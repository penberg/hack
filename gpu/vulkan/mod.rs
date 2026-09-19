//! The model on a GPU, through Vulkan compute: every operation is a
//! hand-written kernel in `kernels/`, compiled to SPIR-V at build time.
//!
//! Operations are recorded into a command buffer as they are called, with a
//! barrier between each, and submitted every so many of them without
//! waiting, from a ring of command buffers, so that the GPU works on the
//! first kernels of a forward pass while the CPU records the rest. Reading
//! a buffer back waits for everything submitted.

use std::{error::Error, io::Cursor, slice, sync::Mutex};

use ash::{
    khr::{push_descriptor, shader_float16_int8},
    vk,
};
use rayon::prelude::*;

use crate::{
    CONV_KERNEL, Device, HADAMARD_BLOCK, Tensor,
    ternary::{BLOCK, BLOCK_BYTES},
};

/// Size of each of the two staging buffers host memory is copied to and
/// from the device through: uploads alternate between them, so that one is
/// filled while the device copies from the other.
const STAGING: usize = 64 << 20;

/// Rows of ternary weights one workgroup of the single-token and batch
/// matmul kernels takes.
const TERNARY_ROWS: usize = 8;
/// Rows and tokens of a tile of `matmul_ternary_tile.wgsl`; a batch of at
/// least half a tile's tokens goes through it.
const TERNARY_TILE_ROWS: usize = 64;
const TERNARY_TILE_TOKENS: usize = 64;
/// Workgroup memory the tile kernel takes: the rows' and the tokens' block
/// as half floats.
const TERNARY_TILE_MEMORY: usize = (TERNARY_TILE_ROWS + TERNARY_TILE_TOKENS) * 2 * 128;
/// Most tokens the kernel for a few tokens takes at once.
const TERNARY_FEW_TOKENS: usize = 8;
/// Most words of packed activations a batch through the tile kernel or the
/// kernel for a few tokens has, two tokens to a word: the size of the
/// buffer they are packed into.
const PACKED: usize = 1 << 20;

/// Command buffers in the ring, and kernels recorded into one before it is
/// submitted.
const RING: usize = 4;
const SUBMIT_EVERY: usize = 128;

/// Positions one workgroup of the attention kernel takes.
const CHUNK: usize = 128;

/// Most floats of attention partials one dispatch writes, one chunk's
/// maximum, sum, and weighted values per head of each token: the size of
/// the partials buffer. Longer batches are dispatched a few tokens at a
/// time.
const PARTIALS: usize = 16 << 20;

/// A vector of f32 activations in device memory.
pub struct Buffer {
    buf: vk::Buffer,
    len: usize,
    cap: usize,
}

/// A cache of f16 activations in device memory, two to a 32-bit word.
pub struct Cache {
    buf: vk::Buffer,
    len: usize,
}

/// A weight matrix in device memory, bf16 or ternary.
pub struct Weight {
    buf: vk::Buffer,
    shape: Vec<usize>,
    ternary: bool,
}

struct Kernels {
    matmul: vk::Pipeline,
    matmul_ternary: vk::Pipeline,
    matmul_ternary_batch: vk::Pipeline,
    matmul_ternary_tile: vk::Pipeline,
    matmul_ternary_few: vk::Pipeline,
    pack_halves: vk::Pipeline,
    add: vk::Pipeline,
    rmsnorm: vk::Pipeline,
    l2norm: vk::Pipeline,
    rope: vk::Pipeline,
    attention: vk::Pipeline,
    attention_combine: vk::Pipeline,
    silu_mul: vk::Pipeline,
    sigmoid_mul: vk::Pipeline,
    store: vk::Pipeline,
    hadamard: vk::Pipeline,
    norm_rotate: vk::Pipeline,
    conv: vk::Pipeline,
    delta_net: vk::Pipeline,
}

impl Kernels {
    fn all(&self) -> [vk::Pipeline; 19] {
        [
            self.matmul,
            self.matmul_ternary,
            self.matmul_ternary_batch,
            self.matmul_ternary_tile,
            self.matmul_ternary_few,
            self.pack_halves,
            self.add,
            self.rmsnorm,
            self.l2norm,
            self.rope,
            self.attention,
            self.attention_combine,
            self.silu_mul,
            self.sigmoid_mul,
            self.store,
            self.hadamard,
            self.norm_rotate,
            self.conv,
            self.delta_net,
        ]
    }
}

/// A host-visible buffer, mapped for as long as the device lives.
struct Mapped {
    buf: vk::Buffer,
    ptr: *mut u8,
}

pub struct Vulkan {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    push: push_descriptor::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    kernels: Kernels,
    memory_types: vk::PhysicalDeviceMemoryProperties,
    staging: [Mapped; 2],
    partials: vk::Buffer,
    /// A batch's activations as half floats, for the tile kernel.
    packed: vk::Buffer,
    max_groups: u32,
    name: String,
    /// Every buffer and its memory, freed when the device is dropped.
    allocations: Mutex<Vec<(vk::Buffer, vk::DeviceMemory)>>,
    ring: Mutex<Ring>,
}

/// Where recording and submission stand in the ring of command buffers.
struct Ring {
    /// The command buffer being recorded into, or to record into next.
    current: usize,
    /// Whether it is open with commands not yet submitted.
    recording: bool,
    /// Which command buffers are submitted and not yet waited for.
    in_flight: [bool; RING],
    /// Kernels recorded into the current command buffer.
    recorded: usize,
    /// The command buffer of the ring whose submission reads each staging
    /// buffer, if one may still be running, and which buffer the next
    /// upload fills.
    staging_slot: [Option<usize>; 2],
    staging_next: usize,
}

// The mapped pointers are only used from whichever thread owns the device.
unsafe impl Send for Vulkan {}

impl Vulkan {
    /// Opens the first GPU Vulkan finds, preferring a discrete one.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        unsafe {
            let entry = ash::Entry::load()?;
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
            let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)?;

            let physical = instance
                .enumerate_physical_devices()?
                .into_iter()
                .min_by_key(|&pd| match instance.get_physical_device_properties(pd).device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    _ => 3,
                })
                .ok_or("no Vulkan device")?;
            let props = instance.get_physical_device_properties(physical);
            let name = props.device_name_as_c_str()?.to_string_lossy().into_owned();
            if props.api_version < vk::API_VERSION_1_1 {
                return Err(format!("{name} does not support Vulkan 1.1").into());
            }
            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
            instance.get_physical_device_properties2(physical, &mut props2);
            let needed = vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC | vk::SubgroupFeatureFlags::SHUFFLE;
            if !subgroup.supported_operations.contains(needed) || subgroup.subgroup_size < 16 {
                return Err(format!("{name} lacks the subgroup operations the kernels use").into());
            }
            if (props.limits.max_compute_shared_memory_size as usize) < TERNARY_TILE_MEMORY {
                return Err(format!("{name} has less than the {} KB of workgroup memory the kernels use", TERNARY_TILE_MEMORY / 1024).into());
            }
            let extensions = instance.enumerate_device_extension_properties(physical)?;
            for needed in [push_descriptor::NAME, shader_float16_int8::NAME] {
                if !extensions.iter().any(|ext| ext.extension_name_as_c_str() == Ok(needed)) {
                    return Err(format!("{name} lacks {}", needed.to_string_lossy()).into());
                }
            }
            let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut float16);
            instance.get_physical_device_features2(physical, &mut features);
            if float16.shader_float16 == 0 {
                return Err(format!("{name} lacks 16-bit float arithmetic, which the kernels use").into());
            }

            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("no compute queue")? as u32;
            let queue_info = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&[1.0])];
            let extension_names = [push_descriptor::NAME.as_ptr(), shader_float16_int8::NAME.as_ptr()];
            let mut float16 = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extension_names)
                .push_next(&mut float16);
            let device = instance.create_device(physical, &device_info, None)?;
            let push = push_descriptor::Device::new(&instance, &device);
            let queue = device.get_device_queue(family, 0);

            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let cmds = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(RING as u32),
            )?;
            let mut fences = Vec::with_capacity(RING);
            for _ in 0..RING {
                fences.push(device.create_fence(&vk::FenceCreateInfo::default(), None)?);
            }

            // Every kernel binds up to eight storage buffers, pushed with each
            // dispatch, and takes its sizes as push constants.
            let bindings: Vec<_> = (0..8)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                    .bindings(&bindings),
                None,
            )?;
            let set_layouts = [set_layout];
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(32)];
            let layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_range),
                None,
            )?;

            let kernel = |spv: &[u8]| -> Result<vk::Pipeline, Box<dyn Error>> {
                let words = ash::util::read_spv(&mut Cursor::new(spv))?;
                let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
                let stage = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main");
                let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
                let pipeline = device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                    .map_err(|(_, e)| e)?[0];
                device.destroy_shader_module(module, None);
                Ok(pipeline)
            };
            macro_rules! spv {
                ($name:literal) => {
                    kernel(include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv")))?
                };
            }
            let memory_types = instance.get_physical_device_memory_properties(physical);
            let kernels = Kernels {
                matmul: spv!("matmul"),
                matmul_ternary: spv!("matmul_ternary"),
                matmul_ternary_batch: spv!("matmul_ternary_batch"),
                matmul_ternary_tile: spv!("matmul_ternary_tile"),
                matmul_ternary_few: spv!("matmul_ternary_few"),
                pack_halves: spv!("pack_halves"),
                add: spv!("add"),
                rmsnorm: spv!("rmsnorm"),
                l2norm: spv!("l2norm"),
                rope: spv!("rope"),
                attention: spv!("attention"),
                attention_combine: spv!("attention_combine"),
                silu_mul: spv!("silu_mul"),
                sigmoid_mul: spv!("sigmoid_mul"),
                store: spv!("store"),
                hadamard: spv!("hadamard"),
                norm_rotate: spv!("norm_rotate"),
                conv: spv!("conv"),
                delta_net: spv!("delta_net"),
            };

            let mut gpu = Self {
                _entry: entry,
                instance,
                device,
                push,
                queue,
                pool,
                cmds,
                fences,
                set_layout,
                layout,
                kernels,
                memory_types,
                staging: [
                    Mapped {
                        buf: vk::Buffer::null(),
                        ptr: std::ptr::null_mut(),
                    },
                    Mapped {
                        buf: vk::Buffer::null(),
                        ptr: std::ptr::null_mut(),
                    },
                ],
                partials: vk::Buffer::null(),
                packed: vk::Buffer::null(),
                max_groups: props.limits.max_compute_work_group_count[0],
                name,
                allocations: Mutex::new(Vec::new()),
                ring: Mutex::new(Ring {
                    current: 0,
                    recording: false,
                    in_flight: [false; RING],
                    recorded: 0,
                    staging_slot: [None; 2],
                    staging_next: 0,
                }),
            };
            gpu.staging = [gpu.map(STAGING)?, gpu.map(STAGING)?];
            gpu.partials = gpu.buffer(PARTIALS * 4, vk::MemoryPropertyFlags::DEVICE_LOCAL)?.0;
            gpu.packed = gpu.buffer(PACKED * 4, vk::MemoryPropertyFlags::DEVICE_LOCAL)?.0;
            Ok(gpu)
        }
    }

    /// Name of the GPU.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Creates a buffer of `bytes` in memory of the given kind.
    fn buffer(&self, bytes: usize, flags: vk::MemoryPropertyFlags) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn Error>> {
        unsafe {
            let info = vk::BufferCreateInfo::default()
                .size(bytes.max(4) as u64)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&info, None)?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let types = &self.memory_types.memory_types[..self.memory_types.memory_type_count as usize];
            let index = types
                .iter()
                .enumerate()
                .position(|(i, t)| req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(flags))
                .ok_or("no suitable memory type")? as u32;
            let mem = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )?;
            self.device.bind_buffer_memory(buf, mem, 0)?;
            self.allocations.lock().unwrap().push((buf, mem));
            Ok((buf, mem))
        }
    }

    /// Creates a host-visible buffer and maps it.
    fn map(&self, bytes: usize) -> Result<Mapped, Box<dyn Error>> {
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let (buf, mem) = self.buffer(bytes, flags)?;
        let ptr = unsafe { self.device.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? };
        Ok(Mapped { buf, ptr: ptr.cast() })
    }

    /// Creates a buffer in device memory.
    fn device_buffer(&self, bytes: usize) -> vk::Buffer {
        self.buffer(bytes, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .expect("out of GPU memory")
            .0
    }

    /// The command buffer open for recording: the current one of the ring,
    /// begun if it is not open, once the GPU is done with it.
    fn cmd(&self) -> vk::CommandBuffer {
        let mut ring = self.ring.lock().unwrap();
        let current = ring.current;
        let cmd = self.cmds[current];
        if !ring.recording {
            if ring.in_flight[current] {
                self.wait(&mut ring, current);
            }
            unsafe {
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .unwrap();
                self.device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .unwrap();
            }
            ring.recording = true;
            ring.recorded = 0;
        }
        cmd
    }

    /// Waits for the GPU to finish command buffer `i` of the ring.
    fn wait(&self, ring: &mut Ring, i: usize) {
        unsafe {
            self.device.wait_for_fences(&[self.fences[i]], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fences[i]]).unwrap();
        }
        ring.in_flight[i] = false;
    }

    /// Makes everything recorded so far visible to everything recorded next.
    fn barrier(&self, cmd: vk::CommandBuffer) {
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe {
            self.device
                .cmd_pipeline_barrier(cmd, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]);
        }
    }

    /// Submits what is recorded, without waiting, and moves on to the next
    /// command buffer of the ring.
    fn submit(&self) {
        let mut ring = self.ring.lock().unwrap();
        if !ring.recording {
            return;
        }
        let i = ring.current;
        unsafe {
            self.device.end_command_buffer(self.cmds[i]).unwrap();
            let cmds = [self.cmds[i]];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            self.device.queue_submit(self.queue, &[submit], self.fences[i]).unwrap();
        }
        ring.in_flight[i] = true;
        ring.recording = false;
        ring.current = (i + 1) % RING;
    }

    /// Submits everything recorded and waits for all of it to finish.
    fn flush(&self) {
        self.submit();
        let mut ring = self.ring.lock().unwrap();
        for i in 0..RING {
            if ring.in_flight[i] {
                self.wait(&mut ring, i);
            }
        }
        ring.staging_slot = [None; 2];
    }

    /// Records a kernel over `groups` workgroups, bound to `buffers` in
    /// order, with `params` as its push constants.
    fn dispatch<P: Copy>(&self, pipeline: vk::Pipeline, buffers: &[vk::Buffer], params: &P, groups: (u32, u32)) {
        let cmd = self.cmd();
        // A kernel binds at most eight buffers; a thousand dispatches a token
        // is worth not allocating for.
        assert!(buffers.len() <= 8);
        let mut infos = [vk::DescriptorBufferInfo::default(); 8];
        for (info, &buf) in infos.iter_mut().zip(buffers) {
            *info = vk::DescriptorBufferInfo::default().buffer(buf).offset(0).range(vk::WHOLE_SIZE);
        }
        let mut writes = [vk::WriteDescriptorSet::default(); 8];
        for (i, (write, info)) in writes.iter_mut().zip(&infos).enumerate() {
            *write = vk::WriteDescriptorSet::default()
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info));
        }
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.push
                .cmd_push_descriptor_set(cmd, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &writes[..buffers.len()]);
            self.device
                .cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytes(params));
            self.device.cmd_dispatch(cmd, groups.0, groups.1, 1);
        }
        self.barrier(cmd);
        // Enough recorded for the GPU to get going on.
        let recorded = {
            let mut ring = self.ring.lock().unwrap();
            ring.recorded += 1;
            ring.recorded
        };
        if recorded >= SUBMIT_EVERY {
            self.submit();
        }
    }

    /// Workgroups for `count` items of work in a one-dimensional kernel.
    fn groups(&self, count: usize, per_group: usize) -> (u32, u32) {
        let groups = count.div_ceil(per_group);
        assert!(groups <= self.max_groups as usize, "{groups} workgroups is more than the GPU allows");
        (groups as u32, 1)
    }

    /// Copies bytes from the host into a device buffer, through staging.
    ///
    /// Each chunk of a staging buffer's size is copied into the staging
    /// buffer the last upload did not use, once whatever last read that one
    /// has finished, and its copy to the device is submitted at once behind
    /// whatever is pending, so that the device copies one chunk while the
    /// host fills the other.
    fn upload_bytes(&self, dst: vk::Buffer, offset: usize, data: &[u8]) {
        for (i, chunk) in data.chunks(STAGING).enumerate() {
            let which = {
                let mut ring = self.ring.lock().unwrap();
                let which = ring.staging_next;
                if let Some(slot) = ring.staging_slot[which].take()
                    && ring.in_flight[slot]
                {
                    self.wait(&mut ring, slot);
                }
                which
            };
            unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), self.staging[which].ptr, chunk.len()) };
            let cmd = self.cmd();
            let region = vk::BufferCopy::default()
                .dst_offset((offset + i * STAGING) as u64)
                .size(chunk.len() as u64);
            unsafe { self.device.cmd_copy_buffer(cmd, self.staging[which].buf, dst, &[region]) };
            self.barrier(cmd);
            let slot = self.ring.lock().unwrap().current;
            self.submit();
            let mut ring = self.ring.lock().unwrap();
            ring.staging_slot[which] = Some(slot);
            ring.staging_next = 1 - which;
        }
    }

    /// Copies bytes from a device buffer to the host, through staging: the
    /// copy is recorded behind whatever is pending, and everything is
    /// submitted together.
    fn download_bytes(&self, src: vk::Buffer, data: &mut [u8]) {
        for (i, chunk) in data.chunks_mut(STAGING).enumerate() {
            unsafe {
                let cmd = self.cmd();
                let region = vk::BufferCopy::default()
                    .src_offset((i * STAGING) as u64)
                    .size(chunk.len() as u64);
                self.device.cmd_copy_buffer(cmd, src, self.staging[0].buf, &[region]);
                self.barrier(cmd);
                self.flush();
                std::ptr::copy_nonoverlapping(self.staging[0].ptr, chunk.as_mut_ptr(), chunk.len());
            }
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &fence in &self.fences {
                self.device.destroy_fence(fence, None);
            }
            for pipeline in self.kernels.all() {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_command_pool(self.pool, None);
            for (buf, mem) in self.allocations.lock().unwrap().drain(..) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulParams {
    rows: u32,
    cols: u32,
    n: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RmsnormParams {
    dim: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    rot_dim: u32,
    pos: u32,
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HadamardParams {
    width: u32,
    inverse: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    n: u32,
    channels: u32,
    q_dim: u32,
    k_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaNetParams {
    n: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    head_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackParams {
    n: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttentionParams {
    n_heads: u32,
    head_dim: u32,
    n_kv_heads: u32,
    pos: u32,
    first: u32,
    chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct StoreParams {
    offset: u32,
    len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LenParams {
    len: u32,
}

impl Device for Vulkan {
    type Buffer = Buffer;
    type Weight = Weight;
    type Cache = Cache;

    fn upload(&self, tensor: Tensor) -> Weight {
        match tensor {
            Tensor::Bf16 { shape, data } => {
                assert!(data.len().is_multiple_of(8), "weights must come in multiples of eight");
                let buf = self.device_buffer(data.len() * 2);
                // bf16 bits, two to a 32-bit word, in memory order: the
                // kernels take the low half of a word as the first weight,
                // as little-endian does.
                let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
                self.upload_bytes(buf, 0, bytes);
                Weight {
                    buf,
                    shape,
                    ternary: false,
                }
            }
            Tensor::Ternary { shape, data } => {
                assert_eq!(data.len(), shape[0] * crate::ternary::row_bytes(shape[1]));
                let data = block_major(&data, shape[0], shape[1]);
                let buf = self.device_buffer(data.len());
                self.upload_bytes(buf, 0, &data);
                Weight {
                    buf,
                    shape,
                    ternary: true,
                }
            }
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        let buf = self.device_buffer(len * 4);
        let cmd = self.cmd();
        unsafe { self.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0) };
        self.barrier(cmd);
        Buffer { buf, len, cap: len }
    }

    fn alloc_cache(&self, len: usize) -> Cache {
        let buf = self.device_buffer(len.div_ceil(2) * 4);
        let cmd = self.cmd();
        unsafe { self.device.cmd_fill_buffer(cmd, buf, 0, vk::WHOLE_SIZE, 0) };
        self.barrier(cmd);
        Cache { buf, len }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(len <= buf.cap, "a buffer of {} activations can't hold {len}", buf.cap);
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        let mut out = vec![0.0f32; buf.len];
        let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), buf.len * 4) };
        self.download_bytes(buf.buf, bytes);
        out
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(buf.len, data.len());
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        self.upload_bytes(buf.buf, 0, bytes);
    }

    fn copy(&self, dst: &mut Buffer, dst_offset: usize, src: &Buffer, src_offset: usize, len: usize) {
        assert!(dst_offset + len <= dst.len && src_offset + len <= src.len);
        let cmd = self.cmd();
        let region = vk::BufferCopy::default()
            .src_offset((src_offset * 4) as u64)
            .dst_offset((dst_offset * 4) as u64)
            .size((len * 4) as u64);
        unsafe { self.device.cmd_copy_buffer(cmd, src.buf, dst.buf, &[region]) };
        self.barrier(cmd);
    }

    fn store(&self, cache: &mut Cache, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= cache.len);
        // The kernel packs whole words of two activations.
        assert!(offset.is_multiple_of(2) && src.len.is_multiple_of(2));
        let params = StoreParams {
            offset: offset as u32,
            len: src.len as u32,
        };
        let groups = self.groups(src.len / 2, 256);
        self.dispatch(self.kernels.store, &[cache.buf, src.buf], &params, groups);
    }

    fn read_cache(&self, cache: &Cache, len: usize) -> Vec<u16> {
        assert!(len <= cache.len);
        let mut out = vec![0u16; len];
        let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), len * 2) };
        self.download_bytes(cache.buf, bytes);
        out
    }

    fn write_cache(&self, cache: &mut Cache, data: &[u16]) {
        assert!(data.len() <= cache.len);
        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
        self.upload_bytes(cache.buf, 0, bytes);
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        assert!(cols % 8 == 0);
        // A batch of tokens through ternary weights multiplies the
        // activations packed as half floats: a few tokens by the kernel
        // that reads the weights once for up to eight, and many as a tiled
        // matrix product, a workgroup per tile of rows and tokens, the row
        // tiles across the grid and down it as wide as the GPU allows, and
        // each row of the grid repeated for every tile of tokens.
        let few = w.ternary && (2..=TERNARY_FEW_TOKENS).contains(&n);
        if few || (w.ternary && n >= TERNARY_TILE_TOKENS / 2) {
            let pairs = n.div_ceil(2).next_multiple_of(4);
            assert!(pairs * cols <= PACKED, "a batch of {n} tokens of {cols} is more than the tile kernel packs");
            let params = PackParams {
                n: n as u32,
                cols: cols as u32,
            };
            let groups = (cols.div_ceil(256) as u32, pairs as u32);
            self.dispatch(self.kernels.pack_halves, &[self.packed, x.buf], &params, groups);
        }
        if few {
            let count = rows.div_ceil(TERNARY_ROWS);
            let width = count.min(self.max_groups as usize);
            let params = MatmulParams {
                rows: rows as u32,
                cols: cols as u32,
                n: n as u32,
                stride: width as u32,
            };
            self.dispatch(self.kernels.matmul_ternary_few, &[out.buf, w.buf, self.packed], &params, (width as u32, count.div_ceil(width) as u32));
            return;
        }
        if w.ternary && n >= TERNARY_TILE_TOKENS / 2 {
            let count = rows.div_ceil(TERNARY_TILE_ROWS);
            let width = count.min(self.max_groups as usize);
            let height = count.div_ceil(width) * n.div_ceil(TERNARY_TILE_TOKENS);
            assert!(height <= self.max_groups as usize, "{height} rows of workgroups is more than the GPU allows");
            let params = MatmulParams {
                rows: rows as u32,
                cols: cols as u32,
                n: n as u32,
                stride: width as u32,
            };
            self.dispatch(self.kernels.matmul_ternary_tile, &[out.buf, w.buf, self.packed], &params, (width as u32, height as u32));
            return;
        }
        // One workgroup per row, or per eight rows of ternary weights, in a
        // grid as wide as the GPU allows. A few tokens are worth unpacking
        // the ternary weights once for several.
        let (kernel, per_group) = if w.ternary && n > 1 {
            (self.kernels.matmul_ternary_batch, TERNARY_ROWS)
        } else if w.ternary {
            (self.kernels.matmul_ternary, TERNARY_ROWS)
        } else {
            (self.kernels.matmul, 1)
        };
        let count = rows.div_ceil(per_group);
        let width = count.min(self.max_groups as usize);
        let groups = (width as u32, count.div_ceil(width) as u32);
        let params = MatmulParams {
            rows: rows as u32,
            cols: cols as u32,
            n: n as u32,
            stride: width as u32,
        };
        self.dispatch(kernel, &[out.buf, w.buf, x.buf], &params, groups);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        assert_eq!(x.len, y.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(self.kernels.add, &[x.buf, y.buf], &params, groups);
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Buffer, eps: f32) {
        let dim = weight.len;
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams { dim: dim as u32, eps };
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.rmsnorm, &[x.buf, weight.buf], &params, groups);
    }

    fn l2norm(&self, x: &mut Buffer, dim: usize, eps: f32) {
        assert_eq!(x.len % dim, 0);
        let params = RmsnormParams { dim: dim as u32, eps };
        let groups = self.groups(x.len / dim, 1);
        self.dispatch(self.kernels.l2norm, &[x.buf], &params, groups);
    }

    fn rope(&self, x: &mut Buffer, table: &Buffer, pos: usize, n_heads: usize, head_dim: usize, rot_dim: usize) {
        let n = x.len / (n_heads * head_dim);
        assert_eq!(x.len, n * n_heads * head_dim);
        assert!(rot_dim <= head_dim && rot_dim.is_multiple_of(2));
        let params = RopeParams {
            n_heads: n_heads as u32,
            head_dim: head_dim as u32,
            rot_dim: rot_dim as u32,
            pos: pos as u32,
            n: n as u32,
        };
        let groups = self.groups(n * n_heads * rot_dim / 2, 256);
        self.dispatch(self.kernels.rope, &[x.buf, table.buf], &params, groups);
    }

    fn attention(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Cache,
        v_cache: &Cache,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        let n = q.len / (n_heads * head_dim);
        assert_eq!(q.len, n * n_heads * head_dim);
        assert_eq!(out.len, q.len);
        assert!(head_dim.is_multiple_of(4) && head_dim <= 256);
        assert!(n_heads / n_kv_heads <= 8, "the kernel holds up to eight query heads per key/value head");
        // Every token's heads get as many chunks as the last token attends
        // over, as many tokens to a dispatch as the partials fit.
        let chunks = (pos + n).div_ceil(CHUNK);
        let per_token = n_heads * chunks * (head_dim + 2);
        assert!(per_token <= PARTIALS, "attention over {} positions needs more partials than {PARTIALS}", pos + n);
        let per_dispatch = PARTIALS / per_token;
        for first in (0..n).step_by(per_dispatch) {
            let tokens = per_dispatch.min(n - first);
            let params = AttentionParams {
                n_heads: n_heads as u32,
                head_dim: head_dim as u32,
                n_kv_heads: n_kv_heads as u32,
                pos: pos as u32,
                first: first as u32,
                chunks: chunks as u32,
            };
            let groups = self.groups(tokens * n_kv_heads * chunks, 1);
            self.dispatch(
                self.kernels.attention,
                &[out.buf, q.buf, k_cache.buf, v_cache.buf, self.partials],
                &params,
                groups,
            );
            if chunks > 1 {
                let groups = self.groups(tokens * n_heads, 1);
                self.dispatch(self.kernels.attention_combine, &[out.buf, self.partials], &params, groups);
            }
        }
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        assert_eq!(gate.len, up.len);
        let params = LenParams { len: gate.len as u32 };
        let groups = self.groups(gate.len, 256);
        self.dispatch(self.kernels.silu_mul, &[gate.buf, up.buf], &params, groups);
    }

    fn sigmoid_mul(&self, x: &mut Buffer, gate: &Buffer) {
        assert_eq!(x.len, gate.len);
        let params = LenParams { len: x.len as u32 };
        let groups = self.groups(x.len, 256);
        self.dispatch(self.kernels.sigmoid_mul, &[x.buf, gate.buf], &params, groups);
    }

    fn hadamard(&self, x: &mut Buffer, signs: &Buffer, inverse: bool) {
        let width = signs.len;
        assert!(x.len.is_multiple_of(width) && width.is_multiple_of(HADAMARD_BLOCK));
        let params = HadamardParams {
            width: width as u32,
            inverse: inverse as u32,
        };
        let groups = self.groups(x.len / HADAMARD_BLOCK, 1);
        self.dispatch(self.kernels.hadamard, &[x.buf, signs.buf], &params, groups);
    }

    fn norm_rotate(&self, out: &mut Buffer, x: &Buffer, weight: &Buffer, signs: &Buffer, eps: f32) {
        let width = signs.len;
        assert!(x.len.is_multiple_of(width) && width.is_multiple_of(HADAMARD_BLOCK));
        assert!(out.len == x.len && weight.len == width);
        let params = RmsnormParams { dim: width as u32, eps };
        let groups = self.groups(x.len / HADAMARD_BLOCK, 1);
        self.dispatch(self.kernels.norm_rotate, &[out.buf, x.buf, weight.buf, signs.buf], &params, groups);
    }

    fn conv(
        &self,
        q: &mut Buffer,
        k: &mut Buffer,
        v: &mut Buffer,
        state_out: &mut Buffer,
        x: &Buffer,
        state: &Buffer,
        weight: &Buffer,
    ) {
        let channels = weight.len / CONV_KERNEL;
        let n = x.len / channels;
        assert_eq!(x.len, n * channels);
        assert!(state.len == (CONV_KERNEL - 1) * channels && state_out.len == state.len);
        let (q_dim, k_dim, v_dim) = (q.len / n, k.len / n, v.len / n);
        assert_eq!(q_dim + k_dim + v_dim, channels);
        let params = ConvParams {
            n: n as u32,
            channels: channels as u32,
            q_dim: q_dim as u32,
            k_dim: k_dim as u32,
        };
        let groups = self.groups(n * channels, 256);
        let buffers = [q.buf, k.buf, v.buf, state_out.buf, x.buf, state.buf, weight.buf];
        self.dispatch(self.kernels.conv, &buffers, &params, groups);
    }

    fn delta_net(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        gates: &Buffer,
        decay: &Buffer,
        state: &mut Buffer,
        n_k_heads: usize,
        n_v_heads: usize,
        head_dim: usize,
    ) {
        let n = v.len / (n_v_heads * head_dim);
        assert_eq!(v.len, n * n_v_heads * head_dim);
        assert!(q.len == n * n_k_heads * head_dim && k.len == q.len && out.len == v.len);
        assert!(gates.len == n * 2 * n_v_heads && decay.len == 2 * n_v_heads);
        assert_eq!(state.len, n_v_heads * head_dim * head_dim);
        assert_eq!(head_dim, 128, "the kernel is written for 128-wide heads");
        let params = DeltaNetParams {
            n: n as u32,
            n_k_heads: n_k_heads as u32,
            n_v_heads: n_v_heads as u32,
            head_dim: head_dim as u32,
        };
        let buffers = [out.buf, q.buf, k.buf, v.buf, gates.buf, decay.buf, state.buf];
        self.dispatch(self.kernels.delta_net, &buffers, &params, self.groups(n_v_heads, 1));
    }
}

/// Reorders rows of ternary blocks into block-major order: every row's
/// first block, then every row's second, and so on, so that the lanes of a
/// kernel that take consecutive rows read consecutive blocks. Eight blocks
/// of a row are read together, for the cache's sake.
fn block_major(data: &[u8], rows: usize, cols: usize) -> Vec<u8> {
    let blocks = cols / BLOCK;
    let mut out = vec![0u8; data.len()];
    out.par_chunks_mut(8 * rows * BLOCK_BYTES).enumerate().for_each(|(i, chunk)| {
        let b0 = i * 8;
        let count = chunk.len() / (rows * BLOCK_BYTES);
        for r in 0..rows {
            let src = &data[(r * blocks + b0) * BLOCK_BYTES..][..count * BLOCK_BYTES];
            for (j, block) in src.chunks_exact(BLOCK_BYTES).enumerate() {
                chunk[(j * rows + r) * BLOCK_BYTES..][..BLOCK_BYTES].copy_from_slice(block);
            }
        }
    });
    out
}

/// The bytes of a plain struct of 32-bit fields, as push constants.
fn bytes<P: Copy>(params: &P) -> &[u8] {
    unsafe { slice::from_raw_parts((params as *const P).cast::<u8>(), size_of::<P>()) }
}

#[cfg(test)]
mod tests {
    check_against_cpu!(super::Vulkan::new());

    /// Times the ternary matmul on the model's shapes; run with `--ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn ternary_matmul_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::tests::ternary_matmul_speed(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the kernels at the model's shapes, for one token and for a
    /// batch of 64; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_speed() {
        match super::Vulkan::new() {
            Ok(gpu) => {
                for n in [1, 64] {
                    eprintln!("batch of {n}:");
                    crate::tests::kernel_speed(&gpu, n);
                }
            }
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    #[test]
    #[ignore]
    fn tile_error() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::tests::tile_error(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }

    /// Times the cost of a dispatch; run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn dispatch_overhead() {
        match super::Vulkan::new() {
            Ok(gpu) => crate::tests::dispatch_overhead(&gpu),
            Err(e) => eprintln!("skipping: {e}"),
        }
    }
}
