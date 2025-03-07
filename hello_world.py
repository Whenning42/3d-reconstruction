# Open the video
# Grab a frame from every second of the video
# Run the pytorch3d reconstruction of the frames

# Seed the scene with randomly initialized triangles as the mesh and random guesses
# for initial camera poses.

# Optimize the mesh and camera poses to minimize the reprojection error of the mesh
# vertices to the image frames

import av
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from pytorch3d.renderer import (
    AmbientLights,
    DirectionalLights,
    FoVPerspectiveCameras,
    Materials,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    look_at_rotation,
)
from pytorch3d.structures import Meshes

TRUNCATE = 40


def load_video(video_path: str) -> torch.Tensor:
    # Load the video
    container = av.open(video_path)
    frames = []

    container.streams.video[0].codec_context.skip_frame = "NONKEY"

    total_frames = container.streams.video[0].frames
    i = 0
    for frame in tqdm(container.decode(video=0), total=total_frames):
        img = pil_to_tensor(frame.to_image())
        frames.append(img)

        i += 1
        if TRUNCATE and i >= TRUNCATE:
            break

    return torch.stack(frames)


# TODO:
# Set up pytorch3d rendering loop
def train(video_path: str, res=256):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    images = load_video(video_path)
    images = F.center_crop(images, min(images.shape[2], images.shape[3]))
    images = F.resize(images, (res, res))
    images = images.to(device)

    raster_settings = RasterizationSettings(
        image_size=res,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    with torch.device(device):
        num_tris = 500_000
        # Sample center points in a shell between radius .3 and 1.5
        # To do this, we sample points in a ball, then map radiuses along the ball from
        # [0, 1] to [.3, 1.5]
        mesh_centers = torch.randn((3, num_tris))
        mesh_rs = torch.sum(mesh_centers**2, dim=0) ** 0.5
        target_rs = mesh_rs * (1.5 - 0.3) + 0.3
        scale_rs = mesh_rs / target_rs
        mesh_centers *= scale_rs
        triangle_offsets = torch.randn((3, 3, num_tris)) * 0.01  # (V, D, N)
        triangle_vertices = triangle_offsets + mesh_centers
        triangle_vertices = torch.permute(triangle_vertices, [2, 0, 1])  # to (N, V, D)
        triangle_vertices = triangle_vertices.reshape(-1, 3)
        indices = torch.arange(triangle_vertices.shape[0]).view(-1, 3)  # (N, 3)
        print(triangle_vertices.shape)
        print(indices.shape)
        triangle_vertices.requires_grad = True
        mesh = Meshes(verts=[triangle_vertices], faces=[indices])
        e_tris = triangle_vertices[None].expand(40, -1, -1)
        e_indices = indices[None].expand(40, -1, -1)
        print(e_tris.shape)

        vert_colors = torch.ones(triangle_vertices.shape, requires_grad=True) * 0.5
        e_vert_colors = vert_colors[None].expand(40, -1, -1)
        textures = TexturesVertex(verts_features=e_vert_colors)

        meshes = Meshes(verts=e_tris, faces=e_indices, textures=textures)
        # TODO: Generate initial mesh vertex colors?

        N = images.shape[0]
        camera_rotations = look_at_rotation(
            torch.zeros(N, 3), at=torch.randn((N, 3)), device=device
        )
        camera_rotations.requires_grad = True
        camera_translations = torch.randn((N, 3), device=device, requires_grad=True)
        cameras = FoVPerspectiveCameras(
            znear=0.003,
            zfar=20,
            fov=82,
            degrees=True,
            R=camera_rotations,
            T=camera_translations,
            device=device,
        )

        lights = AmbientLights(device=device)

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=shader,
        )

        opt = Adam([*cameras.parameters(), *lights.parameters(), triangle_vertices])

    # TODO: Mini-batch cameras and images
    for i in tqdm(range(1000)):
        opt.zero_grad()

        rendered = renderer(meshes)
        rendered = rendered.permute([0, 3, 1, 2])  # (N, H, W, C) to (N, C, H, W)
        rendered = rendered[:, :3]

        print(rendered.shape, images.shape)
        loss = torch.mean((rendered - images) ** 2)
        loss.backward()
        print(i, loss)

        opt.step()


if __name__ == "__main__":
    video_path = "office.mp4"
    train(video_path)
