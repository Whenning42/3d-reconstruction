# Open the video
# Grab a frame from every second of the video
# Run the pytorch3d reconstruction of the frames

# Seed the scene with randomly initialized triangles as the mesh and random guesses
# for initial camera poses.

# Optimize the mesh and camera poses to minimize the reprojection error of the mesh
# vertices to the image frames

import av
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from pytorch3d.renderer import (
    AmbientLights,
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
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

TRUNCATE = 15


def load_video(video_path: str) -> torch.Tensor:
    # Load the video
    container = av.open(video_path)
    frames = []

    container.streams.video[0].codec_context.skip_frame = "NONKEY"

    total_frames = container.streams.video[0].frames
    i = 0
    c = 0
    SKIP = 5
    for frame in tqdm(container.decode(video=0), total=total_frames):
        i += 1

        if i % SKIP != 0:
            continue

        c += 1

        img = pil_to_tensor(frame.to_image())
        frames.append(img)

        if TRUNCATE and c >= TRUNCATE:
            break

    return torch.stack(frames).permute([0, 1, 3, 2])


# TODO:
# Set up pytorch3d rendering loop
def train(video_path: str, res=256):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    images = load_video(video_path)
    images = F.center_crop(images, min(images.shape[2], images.shape[3]))
    images = F.resize(images, (res, res))
    images = images / 256
    images = images.to(device)

    sigma = 1e-4
    raster_settings = RasterizationSettings(
        image_size=res,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
        faces_per_pixel=3,
    )

    N = images.shape[0]
    with torch.device(device):
        num_tris = 500_000
        # Sample center points in a shell between radius .3 and 1.5
        # To do this, we sample points in a ball, then map radiuses along the ball from
        # [0, 1] to [.3, 1.5]
        mesh_centers = torch.randn((3, num_tris))
        mesh_rs = torch.sum(mesh_centers**2, dim=0) ** 0.5
        target_rs = mesh_rs * (1.5 - 0.3) + 0.3
        scale_rs = target_rs / mesh_rs
        mesh_centers *= scale_rs
        triangle_offsets = torch.randn((3, 3, num_tris)) * 0.01  # (V, D, T)
        triangle_vertices = triangle_offsets + mesh_centers
        triangle_vertices = torch.permute(triangle_vertices, [2, 0, 1])  # to (T, V, D)
        triangle_vertices = triangle_vertices.reshape(-1, 3)[None]  # to (N, V, D)
        triangle_vertices = triangle_vertices.repeat(N, 1, 1)
        triangle_vertices.requires_grad = True
        indices = torch.arange(triangle_vertices.shape[1]).view(-1, 3)[
            None
        ]  # (N, F, 3)
        indices = indices.repeat(N, 1, 1)

        vert_colors = torch.rand(triangle_vertices.shape)
        vert_colors.requires_grad = True
        textures = TexturesVertex(verts_features=vert_colors)

        mesh = Meshes(
            verts=triangle_vertices,
            faces=indices,
            textures=textures,
        )

        camera_rotations = look_at_rotation(
            torch.zeros(N, 3), at=torch.randn((N, 3)), device=device
        )
        camera_rotations.requires_grad = True
        camera_translations = torch.randn((N, 3), device=device) * 0.1
        camera_translations.requires_grad = True

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

        opt = Adam(
            [
                # *cameras.parameters(),
                # *lights.parameters(),
                {"params": triangle_vertices, "lr": 0.00001},
                {"params": vert_colors, "lr": 0.005},
            ],
            lr=0.001,
        )

    # TODO: Mini-batch cameras and images
    for n in range(15):
        torchvision.utils.save_image(images[n], f"out_dir/target_{n}.png")

    for i in tqdm(range(1_000)):
        opt.zero_grad()

        # TODO: This copy is maybe wasteful, but I get view of view errors without it.
        textures = TexturesVertex(verts_features=vert_colors)
        mesh = Meshes(
            verts=triangle_vertices,
            faces=indices,
            textures=textures,
        )

        rendered = renderer(mesh)
        rendered = rendered.permute([0, 3, 1, 2])  # (N, H, W, C) to (N, C, H, W)
        rendered = rendered[:, :3]

        loss = torch.mean((rendered - images) ** 2)
        loss.backward()

        if i % 20 == 0:
            print(f"Step: {i}, Loss: {loss.item()}")
            print(
                f"Average vertex distance: {torch.mean(torch.sum(triangle_vertices[0] ** 2, dim=1) ** .5)}\n"
                f"Min vertex distance: {torch.min(torch.sum(triangle_vertices[0] ** 2, dim=1) ** .5)}"
            )

        if i % 100 == 0:
            for n in range(15):
                torchvision.utils.save_image(rendered[n], f"out_dir/{i}_{n}.png")

        opt.step()


if __name__ == "__main__":
    torch.manual_seed(12)
    video_path = "office.mp4"
    train(video_path)
