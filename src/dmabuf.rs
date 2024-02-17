use std::{
    fs::File,
    os::fd::{FromRawFd, IntoRawFd, RawFd},
    sync::Arc,
};

use vulkano::{
    device::Device,
    format::Format,
    image::{
        sys::RawImage, Image, ImageCreateFlags, ImageCreateInfo, ImageTiling, ImageUsage,
        SubresourceLayout,
    },
    memory::{
        allocator::{MemoryAllocator, MemoryTypeFilter},
        DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MemoryAllocateFlags, MemoryAllocateInfo, MemoryImportInfo, MemoryPropertyFlags,
        ResourceMemory,
    },
    sync::Sharing,
};
use wlx_capture::{
    frame::{
        FourCC, DRM_FORMAT_ABGR8888, DRM_FORMAT_ARGB8888, DRM_FORMAT_XBGR8888, DRM_FORMAT_XRGB8888,
    },
    wayland::wayland_client::backend::smallvec::SmallVec,
};

#[cfg(target_os = "linux")]
/// Struct that contains a Linux file descriptor for importing, when creating an image. Since a file descriptor is used for each
/// plane in the case of multiplanar images, each fd needs to have an offset and a row pitch in order to interpret the imported data.
pub struct SubresourceData {
    /// The file descriptor handle of a layer of an image.
    pub fd: RawFd,

    /// The byte offset from the start of the plane where the image subresource begins.
    pub offset: u64,

    ///  Describes the number of bytes between each row of texels in an image plane.
    pub row_pitch: u64,
}

pub fn image_from_dma_buf_fd(
    allocator: &(impl MemoryAllocator + ?Sized),
    device: Arc<Device>,
    extent: [u32; 3],
    format: Format,
    usage: ImageUsage,
    flags: ImageCreateFlags,
    queue_family_indices: impl IntoIterator<Item = u32>,
    mut subresource_data: Vec<SubresourceData>,
    drm_format_modifier: u64,
) -> Option<Arc<Image>> {
    let queue_family_indices: SmallVec<[_; 4]> = queue_family_indices.into_iter().collect();

    // Create a vector of the layout of each image plane.

    // All of the following are automatically true, since the values are explicitly set as such:
    // VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-size-02267
    // VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-arrayPitch-02268
    // VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-depthPitch-02269
    let layout: Vec<SubresourceLayout> = subresource_data
        .iter_mut()
        .map(
            |SubresourceData {
                 fd: _,
                 offset,
                 row_pitch,
             }| {
                SubresourceLayout {
                    offset: *offset,
                    size: 0,
                    row_pitch: *row_pitch,
                    array_pitch: None,
                    depth_pitch: None,
                }
            },
        )
        .collect();

    let fds: Vec<RawFd> = subresource_data
        .iter_mut()
        .map(
            |SubresourceData {
                 fd,
                 offset: _,
                 row_pitch: _,
             }| { *fd },
        )
        .collect();

    let external_memory_handle_types = ExternalMemoryHandleTypes::DMA_BUF;

    let image = RawImage::new(
        device.clone(),
        ImageCreateInfo {
            flags,
            extent,
            format,
            usage,
            sharing: if queue_family_indices.len() >= 2 {
                Sharing::Concurrent(queue_family_indices)
            } else {
                Sharing::Exclusive
            },
            external_memory_handle_types,
            tiling: ImageTiling::DrmFormatModifier,
            drm_format_modifiers: vec![drm_format_modifier],
            drm_format_modifier_plane_layouts: layout,
            ..Default::default()
        },
    )
    .map_err(|err| {
        eprintln!("Error creating image: {:?}", err);
        err
    })
    .ok()?;

    let requirements = image.memory_requirements()[0];
    let memory_type_index = allocator
        .find_memory_type_index(
            requirements.memory_type_bits,
            MemoryTypeFilter {
                required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                ..Default::default()
            },
        )
        .expect("failed to find a suitable memory type");

    assert!(device.enabled_extensions().khr_external_memory_fd);
    assert!(device.enabled_extensions().khr_external_memory);
    assert!(device.enabled_extensions().ext_external_memory_dma_buf);

    let memory = unsafe {
        // Try cloning underlying fd
        let file = File::from_raw_fd(*fds.first().expect("file descriptor Vec is empty"));
        let new_file = file.try_clone().expect("error cloning file descriptor");

        // Turn the original file descriptor back into a raw fd to avoid ownership problems
        file.into_raw_fd();
        DeviceMemory::import(
            device,
            MemoryAllocateInfo {
                allocation_size: requirements.layout.size(),
                memory_type_index,
                dedicated_allocation: Some(DedicatedAllocation::Image(&image)),
                export_handle_types: ExternalMemoryHandleTypes::empty(),
                flags: MemoryAllocateFlags::empty(),
                ..Default::default()
            },
            MemoryImportInfo::Fd {
                handle_type: ExternalMemoryHandleType::DmaBuf,
                file: new_file,
            },
        )
        .unwrap() // TODO: Handle
    };

    let mem_alloc = ResourceMemory::new_dedicated(memory);

    debug_assert!(mem_alloc.offset() % requirements.layout.alignment().as_nonzero() == 0);
    debug_assert!(mem_alloc.size() == requirements.layout.size());

    Some(Arc::new(unsafe {
        image
            .bind_memory_unchecked([mem_alloc])
            .map_err(|err| {
                eprintln!("Error binding image memory: {:?}", err.0);
                err
            })
            .ok()?
    }))
}

pub fn fourcc_to_vk(fourcc: FourCC) -> Format {
    match fourcc.value {
        DRM_FORMAT_ABGR8888 => Format::R8G8B8A8_UNORM,
        DRM_FORMAT_XBGR8888 => Format::R8G8B8A8_UNORM,
        DRM_FORMAT_ARGB8888 => Format::B8G8R8A8_UNORM,
        DRM_FORMAT_XRGB8888 => Format::B8G8R8A8_UNORM,
        _ => panic!("Unsupported memfd format {}", fourcc),
    }
}
