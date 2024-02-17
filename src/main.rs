use std::{error::Error, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
        CommandBufferUsage, RecordingCommandBuffer, RenderPassBeginInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use wlx_capture::{
    frame::{
        DrmFormat, DRM_FORMAT_ABGR8888, DRM_FORMAT_ARGB8888, DRM_FORMAT_XBGR8888,
        DRM_FORMAT_XRGB8888,
    },
    pipewire::{pipewire_select_screen, PipewireCapture},
    WlxCapture,
};

use crate::dmabuf::fourcc_to_vk;

mod dmabuf;
mod receiver;

fn main() -> Result<(), impl Error> {
    //let layers = vec!["VK_LAYER_KHRONOS_validation".to_owned()];
    let layers = vec![];

    let event_loop = EventLoop::new().unwrap();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop).unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            enabled_layers: layers,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        khr_external_memory: true,
        khr_external_memory_fd: true,
        ext_external_memory_dma_buf: true,
        ext_image_drm_format_modifier: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct Vertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    let vertices = [
        Vertex {
            position: [-1., -1.],
        },
        Vertex {
            position: [-1., 1.],
        },
        Vertex {
            position: [1., -1.],
        },
        Vertex { position: [1., 1.] },
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices,
    )
    .unwrap();

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let uploads = RecordingCommandBuffer::new(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferLevel::Primary,
        CommandBufferBeginInfo {
            usage: CommandBufferUsage::OneTimeSubmit,
            ..Default::default()
        },
    )
    .unwrap();

    // upload any additional buffers, textures here

    let allow_dmabuf = std::env::args().any(|arg| arg == "--dmabuf");
    let drm_formats = if !allow_dmabuf {
        println!("Using SHM capture. To use DMA-buf capture, start with --dmabuf");
        vec![]
    } else {
        println!("Using DMA-buf capture. If the screen is blank for you, switch to SHM by starting without --dmabuf");

        let possible_formats = [
            DRM_FORMAT_ABGR8888.into(),
            DRM_FORMAT_XBGR8888.into(),
            DRM_FORMAT_ARGB8888.into(),
            DRM_FORMAT_XRGB8888.into(),
        ];
        let mut final_formats = vec![];

        for &f in &possible_formats {
            let vk_fmt = fourcc_to_vk(f);
            let Ok(props) = device.physical_device().format_properties(vk_fmt) else {
                continue;
            };
            final_formats.push(DrmFormat {
                fourcc: f,
                modifiers: props
                    .drm_format_modifier_properties
                    .iter()
                    // important bit: only allow single-plane
                    .filter(|m| m.drm_format_modifier_plane_count == 1)
                    .map(|m| m.drm_format_modifier)
                    .collect(),
            })
        }
        final_formats
    };

    //TODO
    let token: Option<String> = None;

    let screen = futures::executor::block_on(pipewire_select_screen(token.as_deref())).unwrap();

    let name: Arc<str> = format!("wlx-capture-{}", screen.node_id).into();

    let mut capture = PipewireCapture::new(name.clone(), screen.node_id, 60);
    let rx = capture.init(&drm_formats);
    capture.request_new_frame();

    let mut view;

    loop {
        if let Some(new_view) = crate::receiver::try_receive_frame(
            &rx,
            device.clone(),
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
        ) {
            view = new_view;
            break;
        }
    }

    let physical_size = PhysicalSize::new(view.image().extent()[0], view.image().extent()[1]);
    //window.set_min_inner_size(Some(physical_size));
    //window.set_max_inner_size(Some(physical_size));
    let _ = window.request_inner_size(physical_size);

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format,
                image_extent: [physical_size.width, physical_size.height],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap();

    let pipeline = {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let vertex_input_state = Vertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend::alpha()),
                        ..Default::default()
                    },
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(
        uploads
            .end()
            .unwrap()
            .execute(queue.clone())
            .unwrap()
            .boxed(),
    );

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                elwt.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                if let Some(new_view) = crate::receiver::try_receive_frame(
                    &rx,
                    device.clone(),
                    queue.clone(),
                    memory_allocator.clone(),
                    command_buffer_allocator.clone(),
                ) {
                    view = new_view;
                }

                let sampler = Sampler::new(
                    device.clone(),
                    SamplerCreateInfo {
                        mag_filter: Filter::Linear,
                        min_filter: Filter::Linear,
                        address_mode: [SamplerAddressMode::ClampToEdge; 3],
                        ..Default::default()
                    },
                )
                .unwrap();

                let layout = &pipeline.layout().set_layouts()[0];
                let set = DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    layout.clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        view.clone(),
                        sampler,
                    )],
                    [],
                )
                .unwrap();

                let mut builder = RecordingCommandBuffer::new(
                    command_buffer_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferLevel::Primary,
                    CommandBufferBeginInfo {
                        usage: CommandBufferUsage::OneTimeSubmit,
                        ..Default::default()
                    },
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap();

                unsafe {
                    builder.draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap();
                }

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.end().unwrap();
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => (),
        }
    })
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"#version 310 es
            precision highp float;

            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 tex_coords;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                tex_coords = (position + 1.0) * 0.5;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"#version 310 es
            precision highp float;

            layout(location = 0) in vec2 tex_coords;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform sampler2D in_texture;

            void main() {
                vec4 c = texture(in_texture, tex_coords);
                f_color.rgb = c.rgb;
                f_color.a = min((c.r + c.g + c.b)*100.0, 1.0);
            }
        ",
    }
}
