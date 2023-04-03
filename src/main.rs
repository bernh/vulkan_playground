use std::sync::Arc;
use std::{thread, time};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

use clap::Parser;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    }
}

struct VulkanoComputeApp {
    instance: Arc<Instance>,
    // physical: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    queue_family_index: u32,
}

impl VulkanoComputeApp {
    pub fn initialize() -> Self {
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

        let instance = Instance::new(library, InstanceCreateInfo::default())
            .expect("failed to create instance");

        let physical = instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .next()
            .expect("no devices available");

        let queue_family_index = physical
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.graphics)
            .expect("couldn't find a graphical queue family")
            as u32;

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_external_memory: true,
                    khr_external_memory_fd: true,
                    ..Default::default()
                },
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("failed to create device");
        // extract a single queue from the queues iterator
        let queue = queues.next().unwrap();

        Self {
            instance,
            device,
            queue,
            queue_family_index,
        }
    }

    fn buffer_copy(&self) {
        // this function takes data from the src_buffer and writes results into the dest_buffer

        let source_content: Vec<i32> = (0..64).collect();
        let src_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage {
                transfer_src: true,
                ..Default::default()
            },
            false,
            source_content,
        )
        .expect("failed to create source buffer");

        let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
        let dest_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage {
                transfer_dst: true,
                ..Default::default()
            },
            false,
            destination_content,
        )
        .expect("failed to create destination buffer");

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                src_buffer.clone(),
                dest_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush() // same as signal fence, and then flush
            .unwrap();

        future.wait(None).unwrap(); // None is an optional timeout

        let src_content = src_buffer.read().unwrap();
        let destination_content = dest_buffer.read().unwrap();
        assert_eq!(&*src_content, &*destination_content);

        println!("buffer copy successfully ran!");
    }

    fn compute_shader(&self) {
        // create a buffer with values
        let data_iter = 0..65536;
        let data_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage {
                storage_buffer: true,
                ..Default::default()
            },
            false,
            data_iter,
        )
        .expect("failed to create buffer");

        let shader = cs::load(self.device.clone()).expect("failed to create shader module");

        let compute_pipeline = ComputePipeline::new(
            self.device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("failed to create compute pipeline");

        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0, // 0 is the index of our set
                set,
            )
            .dispatch([1024, 1, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let content = data_buffer.read().unwrap();
        for (n, val) in content.iter().enumerate() {
            assert_eq!(*val, n as u32 * 12);
        }

        println!("compute shader successfully ran!");
    }

    fn inter_process_com_writer(&self) {
        // Simple iterator to construct test data.
        let data = (0..10_000).map(|i| i as f32);

        // Create a CPU accessible buffer initialized with the data.
        let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage {
                transfer_src: true,
                ..BufferUsage::empty()
            }, // Specify this buffer will be used as a transfer source.
            false,
            data,
        )
        .unwrap();

        let device_local_buffer = unsafe {
            DeviceLocalBuffer::<[f32]>::raw_with_exportable_fd(
                self.device.clone(),
                10_000 as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    ..BufferUsage::empty()
                }, // Specify use as a storage buffer and transfer destination.
                self.device.active_queue_family_indices().iter().copied(),
            )
            .expect("Failed to allocate device local buffer")
        };

        let fd = device_local_buffer.export_posix_fd().unwrap();
        println!("File descriptor for GPU memory: {:?}", fd);

        // Create a one-time command to copy between the buffers.
        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                temporary_accessible_buffer,
                device_local_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        // Execute copy command and wait for completion before proceeding.
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        println!("Copied 10000 floats to a GPU buffer");
    }

    fn inter_process_com_reader(&self) {
        // Create a CPU accessible buffer
        let data = (0..10_000).map(|_| 0 as f32);
        let _read_back_buffer = CpuAccessibleBuffer::<[f32]>::from_iter(
            self.device.clone(),
            BufferUsage {
                transfer_dst: true,
                ..BufferUsage::empty()
            },
            false,
            data,
        )
        .unwrap();
    }
}

#[derive(clap::Parser)]
#[command()]
struct Cli {
    #[arg(short, long)]
    reader: bool,
}

fn main() {
    let cli = Cli::parse();

    let app = VulkanoComputeApp::initialize();

    app.buffer_copy();
    app.compute_shader();
    {
        // test GPU shared memory communication between processes
        let extensions = app.device.enabled_extensions();
        if extensions.khr_external_memory && extensions.khr_external_memory_fd {
            println!("Looks like all extensions for external memory handling are available")
        } else {
            panic!("Necessary extensions for external memory handling not available!")
        }

        if cli.reader {
            app.inter_process_com_reader();
        } else {
            app.inter_process_com_writer();
            thread::sleep(time::Duration::from_secs(50));
            println!("done");
        }
    }
}
