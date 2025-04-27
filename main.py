import argparse
import glob
from locale import normalize
import os
from x import visualize_3d_mesh, visualize_3d_vertices
import torch
from W3Z_AnimalSkeletons.smal_torch import SMAL
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
from pytorch3d.io import load_obj
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene
from torchgeometry import rotation_matrix_to_angle_axis
import torchvision
import matplotlib.pyplot as plt
import gc
from W3Z_AnimalTexture.main import SHAPE_TO_ANIMALS
import gdown
import zipfile
from PIL import Image, ImageDraw, ImageFont
from pytorch3d.renderer import (
    MeshRendererWithFragments,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    AmbientLights,
    OpenGLPerspectiveCameras,
    BlendParams
)

from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    PointLights,
    FoVPerspectiveCameras,
)

from CONSTANT import PLABELSNAME, PPOINTINDEX, ANIMAL3DPOINTINDEX, COMBINAPOINTINDEX
import trimesh

def get_point(verts, flag = 'P', visible_verts= None):
    ##### https://github.com/benjiebob/SMALify/smal_fitter/utils.py
    if flag == 'P':
        SELECTPOINTINDEX = PPOINTINDEX
    elif flag == 'ANIMAL3D':
        SELECTPOINTINDEX = ANIMAL3DPOINTINDEX
    elif flag == 'COMBINAPOINTINDEX':
        SELECTPOINTINDEX = COMBINAPOINTINDEX
    else:
        raise ValueError
    num = verts.shape[0]
    if visible_verts is not  None: # [B, V]
        output = []
        for i in range(num):        
            selected_points = torch.stack(
                    [torch.cat([torch.squeeze(verts[i, choice, :], dim=0), visible_verts[i,choice].any().float().view(-1)], dim=-1)
                    if choice.shape[0] == 1 
                    else 
                    torch.cat([verts[i,choice, :].mean(axis=0), visible_verts[i,choice].any().float().view(-1)], dim=-1)
                    for choice in SELECTPOINTINDEX])
            output.append(torch.unsqueeze(selected_points,dim=0)) #[B,V,4]
    else:
        output =[]
        for i in range(num):
            selected_points = torch.stack(
                [torch.squeeze(verts[i,choice, :],dim=0) if choice.shape[0] == 1 else verts[i,choice, :].mean(axis=0) for choice in SELECTPOINTINDEX])
            output.append(torch.unsqueeze(selected_points,dim=0)) #[B,V,3]
            
    final = torch.cat(output,dim=0).type_as(verts)
    return final

def none_to_nan(x):
    if x is None:
        return torch.FloatTensor([float('nan')])
    elif isinstance(x, int):
        return torch.FloatTensor([x])
    else:
        return x

def generate_indices(num_texture, sample_range):
    """
    Generate indices from different ranges with minimum spacing of num_texture
    
    Args:
        num_texture: Minimum spacing between indices
        
    Returns:
        List of generated indices
    """
    # ranges = [(0, 663, 132), (663, 1275, 122), (1275, 1459, 36), (1459, 1590, 26)]
    ranges = []
    for start, end, count in sample_range:
        ranges.append((start * 8 // num_texture, end * 8 // num_texture, count))
    max_index = 12720 // num_texture
    
    result = []
    used_indices = set()
    
    for start, end, count in ranges:
        # Scale the range to account for texture size
        scaled_start = start
        scaled_end = min(end, max_index)
        if scaled_start >= scaled_end:
            print(f"Skipping range [{start}, {end}, {max_index}) because start is greater than or equal to end")
        
        # Generate count unique indices for this range
        attempts = 0
        indices_for_range = []
        while len(indices_for_range) < count and attempts < 1000:
            # Generate a random index in the scaled range
            idx = np.random.randint(scaled_start, scaled_end)
                    
            if idx not in used_indices:
                indices_for_range.append(idx)
                used_indices.add(idx)
            
            attempts += 1
            
        if len(indices_for_range) < count:
            raise ValueError(f"Could not generate {count} indices for range [{start}, {end})")
            
        result.extend(indices_for_range)
    
    # Convert back to original scale
    result = [idx for idx in sorted(result)]
    return result

def download_and_extract():
    """Download zip file from Google Drive and extract it"""
    
    # Google Drive file ID
    # file_id = "1cu7l6MlL_hD8whwJmQUuDtyce5pRXGCV"
    file_id = "1iXGo5FntgnPd3PpgiIZtcx_wWr0CHQMS"
    
    # Create temporary zip file name
    output_zip = "temp_download.zip"
    
    if os.path.exists(output_zip):
        print("Files already exist, skipping download.")
        return
    
    try:
        # Download the file
        print("Downloading file from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_zip, quiet=False)
        
        # Extract the zip file
        print("Extracting zip file...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall("./background")
        
        # Remove the temporary zip file
        # print("Cleaning up...")
        # os.remove(output_zip)
        
        print("Download and extraction completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if os.path.exists(output_zip):
            os.remove(output_zip)

def download_and_extract_background():
    """Download zip file from Google Drive and extract it to background directory"""
    
    # Google Drive file ID
    file_id = "1c7FjVnKslM5G72T97v0RnQPY7hBo15WO"
    
    # Create temporary zip file name
    output_zip = "temp_download_background.zip"
    
    # Define background directory
    background_dir = "./background"
    
    # Check if background directory already exists and has files
    if os.path.exists(background_dir) and os.listdir(background_dir):
        print("Background directory already exists and contains files, skipping download.")
        return
    
    try:
        # Create background directory if it doesn't exist
        os.makedirs(background_dir, exist_ok=True)
        
        # Download the file
        print("Downloading file from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_zip, quiet=False)
        
        # Extract the zip file
        print(f"Extracting zip file to {background_dir}...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            # Get list of files in zip
            file_list = zip_ref.namelist()
            
            # Extract all files
            for file in file_list:
                # Extract only image files
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    zip_ref.extract(file, background_dir)
                    print(f"Extracted: {file}")
        
        # Count extracted files
        extracted_files = len([f for f in os.listdir(background_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Successfully extracted {extracted_files} images to {background_dir}")
        
        # Remove the temporary zip file
        # print("Cleaning up temporary files...")
        # if os.path.exists(output_zip):
        #     os.remove(output_zip)
        #     print(f"Removed temporary zip file: {output_zip}")
        
        print("Download and extraction completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Clean up on error
        if os.path.exists(output_zip):
            os.remove(output_zip)
            print("Cleaned up temporary zip file")
        raise  # R

def init_renderer(camera, shader, image_size, faces_per_pixel):
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=faces_per_pixel) #, bin_size = 50)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings
        ),
        shader=shader
    )
    return renderer

def rotmat_to_axis_angle(x, number = 36):
    ### input [B, J, 3,3 ]
    ### output [B, J*3]
    ###https://github.com/nkolot/SPIN/blob/master/train/trainer.py#L180
    ## may need to check https://github.com/kornia/kornia/pull/1270
    # print(x.shape)
    batch_size = x.shape[0]
    x2 = x.view(-1,3,3)
    # Convert predicted rotation matrices to axis-angle
    pred_rotmat_hom = torch.cat(
        [x2,
         torch.tensor([0, 0, 1], dtype=torch.float32,device=x.device).view(1, 3, 1).expand(batch_size * number, -1, -1)], dim=-1)
    pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
    # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
    pred_pose[torch.isnan(pred_pose)] = 0.0
    return pred_pose

def get_camera_views():
    """Get different camera views"""
    views = {
        'front': torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        'side': torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]),
        'angle': torch.tensor([
            [0.7071, 0, 0.7071],
            [0, 1, 0],
            [-0.7071, 0, 0.7071]
        ]),
        'back': torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]),
    }
    return views

def setup_weak_render_dessie(image_size, faces_per_pixel,device): #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # render the view
    R = torch.tensor([[-1, 0, 0],[0, -1, 0],[0, 0, 1]]).repeat(1, 1, 1).to(device)
    T = torch.zeros(3).repeat(1, 1).to(device)
    fov = 2 * np.arctan(image_size/ (5000. * 2)) * 180 / np.pi
    cameras = OpenGLPerspectiveCameras(zfar=350, fov=fov, R=R, T=T, device=device)
    renderer = init_renderer(cameras,
                             shader=SoftPhongShader(
                                    cameras=cameras,
                                    lights= AmbientLights(device=device),
                                    device=device,
                                    blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color= (1, 1, 1)),
                                ),
                             image_size=image_size,faces_per_pixel=faces_per_pixel,)
    return cameras, renderer

def setup_weak_render(image_size, faces_per_pixel,device, animal_type): #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # render the view
    view_type = 'back'  # or 'side' or 'angle'
    R = get_camera_views()[view_type].repeat(1, 1).to(device)
    # Adjust T to move camera back and slightly up
    if animal_type == 'equidae':
        T = torch.tensor([0, 0, 3.5]).to(device)  # [x, y, z]: z controls distance
    else:
        T = torch.tensor([0, 0, 2.5]).to(device)  # [x, y, z]: z controls distance
    # fov = 2 * np.arctan(image_size/ (5000. * 2)) * 180 / np.pi
    # cameras = OpenGLPerspectiveCameras(
    #     zfar=600, 
    #     fov=fov, 
    #     R=R, 
    #     T=T, 
    #     device=device
    # )
    cameras = FoVPerspectiveCameras(
            device=device,
            R=R[None],  # Add batch dimension: (1, 3, 3)
            T=T[None],  # Add batch dimension: (1, 3)
        )
    
    weak_lights = PointLights(
            device=device,
            location=[[0.0, 2.0, 2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.3, 0.3, 0.3]],
            specular_color=[[0.2, 0.2, 0.2]]
        )

    renderer = init_renderer(cameras,
                             shader=SoftPhongShader(
                                    cameras=cameras,
                                    lights= weak_lights,
                                    device=device,
                                    # blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color= (1, 1, 1)),
                                ),
                             image_size=image_size,faces_per_pixel=faces_per_pixel,)
    return cameras, renderer

@torch.no_grad()
def check_visible_verts_batch(mesh, fragments):
    pix_to_face = fragments.pix_to_face
    # (B, F, 3) where F is the total number of faces across all the meshes in the batch
    packed_faces = mesh._faces_padded #mesh.faces_packed() 
    # (B, V, 3) where V is the total number of verts across all the meshes in the batch
    packed_verts = mesh._verts_padded #mesh.verts_packed() 
    vertex_visibility_map = torch.zeros((packed_verts.shape[0], packed_verts.shape[1])).to(mesh.device) #[B, V]
    for i in range(packed_verts.shape[0]):
        #  Indices of unique visible faces pix_to_face[i].unique()
        # Get Indices of unique visible verts using the vertex indices in the faces packed_faces[i][pix_to_face[i].unique()]
        visible_per_frame = pix_to_face[i].unique()
        visible_per_frame[1:] -= packed_faces.shape[1]*i
        vertex_visibility_map[i, packed_faces[i][visible_per_frame].reshape(-1,).unique()] =1.
    return vertex_visibility_map

@torch.no_grad()
def render_image_mask_and_save_with_preset_render(renderer, cameras, mesh,image_size,
                                   save_intermediate=False, save_dir = ''):  
    
    verts = mesh.verts_packed()  # (V, 3)
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    center = (min_xyz + max_xyz) / 2
    size = (max_xyz - min_xyz)
    # mesh.offset_verts_(-center)


    init_images_tensor, fragments = renderer(mesh) # images
    # np.savez("../scripts/test.npz", image=init_images_tensor)
    
    # get mask
    mask_image_tensor = init_images_tensor[...,[-1]].clone()
    mask_image_tensor[mask_image_tensor > 0] = 1.
    # mask_images, _ = mask_renderer(mesh_mask)
    # obtain visible verts
    visible_verts_tensor = check_visible_verts_batch(mesh, fragments) 
    # obtain kps
    kp_3d_tensor = get_point(mesh._verts_padded, flag = 'P', visible_verts = visible_verts_tensor) #[B,17,4]
    kp_2d_tensor = cameras.transform_points_screen(kp_3d_tensor[:,:,:3], image_size=(image_size,image_size))[:,:,:2] #[N, 17, 2]
    kp_2d_tensor = torch.cat([kp_2d_tensor, kp_3d_tensor[:,:,-1].unsqueeze(2)], dim = 2)
     
    B = init_images_tensor.shape[0]         
    if save_intermediate:  
        init_image = init_images_tensor.permute(0,3,1,2).cpu() #[B, 3, I, I]
        init_image = [transforms.ToPILImage()(init_image[t]).convert("RGB") for t in range(B)]
        kp_2d = kp_2d_tensor.cpu()
        # Draw each keypoint as a circle
        for t in range(B):
            # Create a drawing context
            draw = ImageDraw.Draw(init_image[t])
            for tt,keypoint in enumerate(kp_2d[t]):
                # For a circle, we need the top-left and bottom-right coordinates of the bounding square
                x, y, flag = keypoint
                if flag == 1:
                    r = 2  # radius of the circle
                    draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0, 0))
                    # Draw the text onto the image
                    draw.text((x,y), PLABELSNAME[tt], fill= (0, 0, 255) , font= ImageFont.load_default())

        mask_image = mask_image_tensor[:, :, :, 0].cpu() #[B, I, I, 3]
        mask_image = [transforms.ToPILImage()(mask_image[t]).convert("L") for t in range(B)]
        
        # depth_map = [Image.fromarray(depth_maps_tensor.cpu().numpy()[t]).convert("L")  for t in range(B)]
        # save intermediate results
        for t in range(B):
            init_image[t].save(os.path.join(save_dir, "{}_image_old.png".format(t)))
            mask_image[t].save(os.path.join(save_dir, "{}_mask.png".format(t)))
            # depth_map[t].save(os.path.join(save_dir, "{}_depth_map.png".format(t)))
    return (init_images_tensor,mask_image_tensor, kp_3d_tensor, kp_2d_tensor)

class AnimalPipe(Dataset):
    def __init__(self, args, device, length, FLAG, ANIMAL):
        self.length = length
        self.img_mean = np.array([0.485, 0.456, 0.406])
        self.img_std = np.array([0.229, 0.224, 0.225])
        self.FLAG = FLAG
        self.ANIMAL = ANIMAL
        self.args = args
        self.device = device
        self.image_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        self.image_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std)
        ])
        self.transform = True
        self.get_data()

    def __len__(self):
        return self.length

    def process_image(self, image_tensor):
        img_process = torch.zeros_like(image_tensor)
        for i in range(img_process.shape[0]):
            img = transforms.ToPILImage()(image_tensor[i, ...])
            img = img.convert('RGB') if img.mode != 'RGB' else img
            if self.FLAG == 'TEST':
                img = self.image_transform2(img)
            else:
                img = self.image_transform(img)
            img_process[i] = img
        return img_process
    
    def get_data(self):
        self.get_model()
        self.get_vt_ft()
        self.get_predefinedata()
        self.get_texture()
        self.get_rotation_angle()
        self.get_background()

    def get_model(self):
        self.smal_model = {}
        self.sub_model = {}
        print(f"Getting model for {self.ANIMAL}")
        if self.ANIMAL == 'felidae':
            self.sub_model['felidae_02'] = SMAL(animal_type=self.ANIMAL, animal_subtype='lion',
                                device=self.device)
            self.smal_model[self.ANIMAL] = SMAL(animal_type=self.ANIMAL, animal_subtype='leopard',
                                device=self.device, use_smal_betas=True)
            self.faces = self.smal_model[self.ANIMAL].faces
            # self.num_texture = len(SHAPE_TO_ANIMALS[self.ANIMAL]['felidae_02']) + len(SHAPE_TO_ANIMALS[self.ANIMAL]['felidae_01'])  
        else:
            if self.ANIMAL == "canidae" or self.ANIMAL == "equidae":
                self.smal_model[self.ANIMAL] = SMAL(animal_type=self.ANIMAL,
                                device=self.device)
            else:
                self.smal_model[self.ANIMAL] = SMAL(animal_type=self.ANIMAL,
                                device=self.device, use_smal_betas=True)
            # self.faces = self.smal_model[self.ANIMAL].faces.unsqueeze(0)
            self.faces = self.smal_model[self.ANIMAL].faces
        
        self.num_texture = len(SHAPE_TO_ANIMALS[self.ANIMAL][self.ANIMAL])
    

    def setup_weak_render(self, device, image_size, init_dist=2.2):
        # Get a batch of viewing angles.
        cameras, renderer = setup_weak_render(image_size, faces_per_pixel=1, device=device, animal_type=self.ANIMAL)
        # cameras, renderer = setup_weak_render_dessie(image_size, faces_per_pixel=1, device=device)
        return cameras, renderer

    def get_vt_ft(self):
        entries = SHAPE_TO_ANIMALS[self.ANIMAL]
        if not hasattr(self, "ft") or not hasattr(self, "vt"):
            self.vt = {}
            self.ft = {}

        for entry in entries:
            obj_filename = self.get_obj_filename(f"mesh_{entry}_uvmap")
            smpl_texture_data = load_obj(obj_filename)
            self.vt[entry] = smpl_texture_data[-1].verts_uvs
            self.ft[entry] = smpl_texture_data[1].textures_idx

        # for entry in entries:
        #     print(f"{entry} ft: {self.ft[entry].shape}")
        #     print(f"{entry} vt: {self.vt[entry].shape}")

    def get_predefinedata(self):
        # load data
        pose1, pose2 = self.load_poses()
        '''
        # Define the ranges and the corresponding number of values to sample 
        sampling_ranges = [(0, 663, 132), (663, 1275, 122),  (1275, 1459, 36),  (1459, 1590, 26)]
        random samples some data from the whole data for testing
        '''
        testindex_interval = generate_indices(self.num_texture, [(0, 663, 132), (663, 1275, 122),  (1275, 1459, 36),  (1459, 1590, 26)])
        # testindex_interval = [491, 44, 546, 383, 33, 363, 272, 640, 450, 481, 123, 58, 86, 329, 499, 555, 444, 382, 198,
        #                 193, 515, 566, 430, 378, 642, 288, 434, 605, 99, 467, 645, 274, 498, 453, 380, 381, 365, 314,
        #                 90, 600, 594, 120, 333, 631, 301, 159, 477, 489, 560, 307, 4, 353,
        #                 386, 399, 8, 374, 638, 199, 454, 0, 578, 249, 418, 487, 564, 660, 535, 266, 28, 211, 559, 15,
        #                 591, 603, 313, 178, 536, 571, 441, 108, 361, 254, 277, 597, 526, 25, 335, 129, 217, 143, 45,
        #                 231, 116, 612, 206, 604, 195, 111, 232, 420, 79, 650, 369, 606,
        #                 438, 235, 259, 588, 281, 50, 527, 295, 537, 452, 542, 12, 375, 421, 572, 208, 634, 269, 350,
        #                 167, 628, 502, 26, 244, 540, 617, 342, 590, 1140, 892, 1030, 723, 1011, 876, 687, 912, 1274,
        #                 1112, 750, 1047, 1062, 1144, 950, 698, 921, 913, 1150, 919, 976, 761,
        #                 1231, 795, 1104, 1049, 764, 1260, 781, 1085, 1041, 1036, 1033, 1176, 1242, 1070, 724, 1097,
        #                 682, 1236, 721, 664, 989, 1074, 715, 1268, 819, 680, 1169, 1238, 1263, 738, 994, 1170, 1222,
        #                 1237, 896, 743, 1132, 1053, 807, 1120, 696, 946, 1232, 1068, 1188, 841,
        #                 1196, 673, 1095, 1023, 877, 884, 740, 813, 766, 796, 923, 1044, 741, 792, 987, 1189, 1130,
        #                 1057, 962, 669, 812, 1090, 1055, 1060, 862, 1008, 984, 775, 725, 1014, 929, 980, 940, 1227,
        #                 870, 955, 753, 1185, 712, 770, 1173, 1035, 883, 1077, 998, 991, 791,
        #                 1118, 855, 1184, 1099, 903, 956, 1087, 1398, 1315, 1282, 1343, 1354, 1342, 1369, 1390, 1430,
        #                 1284, 1352, 1393, 1400, 1445, 1327, 1335, 1304, 1295, 1448, 1427, 1421, 1288, 1452, 1416,
        #                 1338, 1339, 1404, 1291, 1329, 1434, 1447, 1319, 1277, 1449, 1323, 1276,
        #                 1528, 1467, 1460, 1551, 1552, 1464, 1462, 1535, 1554, 1559, 1588, 1530, 1520, 1507, 1582,
        #                 1477, 1569, 1543, 1482, 1531, 1544, 1459, 1476, 1512, 1526, 1498]
        testindex = sorted([t * self.num_texture + i for t in testindex_interval for i in range(self.num_texture)])

        if self.FLAG == 'TEST':
            self.pose = torch.from_numpy(np.concatenate([pose1[testindex], pose2[testindex]])).float()
            self.pose_label = np.concatenate([[1] * 132 * self.num_texture, [2] * 122 * self.num_texture, [3] * 36 * self.num_texture, [4] * 26 * self.num_texture, [0] * len(testindex)])
        else:
            '''
            sampling_ranges = [(0, 663, 66), (663, 1275, 61),  (1275, 1459, 18),  (1459, 1590, 13)]
            random samples some data from the whole data for validation
            '''
            validindex_interval = generate_indices(self.num_texture, [(0, 663, 66), (663, 1275, 61),  (1275, 1459, 18),  (1459, 1590, 13)])
            # validindex_interval = [87, 575, 599, 66, 484, 440, 324, 318, 228, 435, 185, 5, 158, 568, 394, 616, 614, 172, 83, 507, 412, 284, 410, 
            #               186, 602, 77, 336, 88, 340, 23, 552, 425, 304, 126, 226, 523, 155, 432, 262, 270, 157, 411, 118, 601, 177, 462,
            #                 302, 456, 113, 181, 511, 184, 424, 52, 22, 377, 152, 423, 161, 49, 264, 188, 530, 236, 426, 229, 842, 1071, 
            #                 966, 1126, 782, 1051, 779, 1177, 829, 1113, 732, 1175, 922, 780, 1168, 1201, 1194, 834, 1179, 793, 822, 1080, 
            #                 1052, 759, 995, 927, 1013, 963, 699, 668, 1241, 1203, 1253, 788, 1020, 996, 772, 677, 836, 1207, 1076, 789, 971, 
            #                 746, 1133, 830, 1114, 756, 722, 949, 900, 777, 979, 1066, 858, 891, 801, 1078, 1101, 704, 1246, 1305, 1414, 1402, 
            #                 1387, 1330, 1366, 1345, 1320, 1285, 1450, 1332, 1302, 1341, 1347, 1326, 1407, 1298, 1287, 1518, 1589, 1532, 1493, 
            #                 1471, 1466, 1461, 1570, 1567, 1560, 1504, 1513, 1577]
            validindex = sorted([t * self.num_texture + i for t in validindex_interval for i in range(self.num_texture)])
            # Find the indices that are not in the subset
            non_subset_indices = np.setdiff1d(np.arange(pose1.shape[0]), validindex+testindex)
            # self.pose_training = torch.from_numpy(np.concatenate([pose1[non_subset_indices], pose2[non_subset_indices]])).float()
            self.pose_training = torch.from_numpy(np.array(pose1[non_subset_indices], dtype=np.float32))
            self.pose_key = non_subset_indices

            # print(f"pose_training.shape: {self.pose_training.shape}")
            # print(f"pose_key.shape: {self.pose_key.shape}")
            # self.pose_valid = torch.from_numpy(np.concatenate([pose1[validindex], pose2[validindex]])).float()
            self.pose_valid = torch.from_numpy(np.array(pose1[validindex], dtype=np.float32))
            self.pose_training_label = np.concatenate([[1] * 465 * self.num_texture, [2] * 429 * self.num_texture, [3] * 130 * self.num_texture, [4] * 92 * self.num_texture, [0] * non_subset_indices.shape[0]])
            self.pose_valid_label = np.concatenate([[1] * 66 * self.num_texture, [2] * 61 * self.num_texture, [3] * 18 * self.num_texture, [4] * 13 * self.num_texture, [0] * len(validindex)])

    def get_texture(self):
        self.texturekey = {}
        self.texture_testindex = {}
        self.texture_trainindex = {}
        self.texture_validindex = {}
        download_and_extract()
        animal_key = SHAPE_TO_ANIMALS[self.ANIMAL]
        if not hasattr(self, "texture"):
            self.texture = {}
            for key in animal_key:
                texture_files = []
                self.texture[key] = []
                for i in range(len(animal_key[key])):
                    for j in range(len(os.listdir(f"./texture/{animal_key[key][i]}"))):
                        texture_files.append(f"./texture/{animal_key[key][i]}/{animal_key[key][i]}_{j}.png")
                
                # print(f"texture_files: {texture_files} and len(texture_files): {len(texture_files)}")
                self.texture[key] = torch.cat([torch.Tensor(
                np.array(Image.open(texture_files[i]).convert("RGB").resize(
                    (self.args.imgsize, self.args.imgsize)), dtype=np.float64) / 255.).unsqueeze(0) for i in range(len(texture_files))],
                                     dim=0)
                
                self.texturekey[key] = [SHAPE_TO_ANIMALS[self.ANIMAL][key][i] for i in range(1, len(SHAPE_TO_ANIMALS[self.ANIMAL][key]))]
                self.texture_testindex[key] = None # !!!!!!!!!!!!!!!!!!! Not implemented
                validindex =  np.random.choice(range(0, len(self.texture[key])), self.num_texture, replace=False) 
                # # Find the indices that are not in the subset
                self.texture_trainindex[key] = np.setdiff1d(np.arange(len(self.texture[key])), validindex)
                self.texture_validindex[key] = np.array(validindex)

    def get_background(self):
        download_and_extract_background()
        if not hasattr(self, "background"):
            self.background_train = sorted(glob.glob(os.path.join('background', '*.jpg')))
            self.background_trainingindex = [i for i in range(len(self.background_train))]
            self.background_transform = transforms.Compose([transforms.Resize((self.args.imgsize, self.args.imgsize)),transforms.ToTensor()])
        
    def get_rotation_angle(self): 
        self.rot_model_4gt = axis_angle_to_matrix(torch.Tensor([np.radians(-90), 0, 0])).unsqueeze(0)


    def get_obj_filename(self, name):
        """Download and get path to the UV map obj file"""
        repo_id = "WatermelonHCMUT/AnimalSkeletons"
        filename = f"{name}.obj"
        
        # Download to a temporary directory or specific location
        obj_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="temp"
        )
        
        return obj_path

    def load_poses(self):
        """Download and load pose files from HuggingFace"""
        repo_id = "WatermelonHCMUT/SamplePose"
        
        # Download poses1.npz
        if self.ANIMAL == 'canidae':
            poses1_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses1_canidae.npz",
                local_dir="./temp"
            )
        elif self.ANIMAL == 'equidae':
            poses1_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses1.npz",
                local_dir="./temp"
            )
        else:
            poses1_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses1_felidae.npz",
                local_dir="./temp"
            )
        
        # Download poses2.npz
        if self.ANIMAL == 'canidae':
            poses2_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses2_canidae.npz",
                local_dir="./temp"
            )
        elif self.ANIMAL == 'equidae':
            poses2_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses2.npz",
                local_dir="./temp"
            )
        else:
            poses2_path = hf_hub_download(
                repo_id=repo_id,
                filename="poses2_felidae.npz",
                local_dir="./temp"
            )
        
        # Load the poses
        pose1 = np.load(poses1_path, allow_pickle=True)['poses']
        pose2 = np.load(poses2_path, allow_pickle=True)['poses'][:12720, :]  # whole
        
        return pose1, pose2
    
    def obtain_SMAL_vertices(self, data_batch_size, poseflag, shapeflag, textureflag, interval=8, cameraindex=0): 
        shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=data_batch_size)
        pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=data_batch_size,
                                                                interval=interval)
        texture_selected, texture_class = self.random_texture(flag=textureflag, data_batch_size=data_batch_size)
        # save_pose_and_texture(pose_selected, f"../scripts/pose_selected_{self.ANIMAL}.npz")

        pose_selected = self.get_pose_gt(pose_selected, cameraindex)
        verts, _, _ = self.smal_model[self.ANIMAL](beta=shape_selected.to(self.device),
                                      theta=pose_selected.to(self.device),
                                      trans=torch.zeros((data_batch_size, 3)).to(self.device))
        trans = self.get_trans_gt(verts, shapeflag)
        verts = verts + trans
        return verts, texture_selected, shapeclass, poseclass, texture_class, shape_selected, pose_selected, trans

    def obtain_SMAL_meshes(self, data_batch_size, verts, texture_selected): 
        assert verts.shape[0] == data_batch_size
        # texture_selected = texture_selected.permute(0, 2, 3, 1)
        a = self.ft[self.ANIMAL].repeat(data_batch_size, 1, 1)
        # save_texture(texture_selected, f"./texture_selected_{self.ANIMAL}.png")
        # print(f"texture_selected.shape: {texture_selected.shape}")
        # print(f"texture_verts.shape: {self.vt[self.ANIMAL].shape}")
        # print(f"verts.shape: {verts.shape} and faces.shape: {self.faces.repeat(verts.shape[0], 1, 1).shape}")
        # verts = transform_vertices(verts)
        textures = TexturesUV(maps=texture_selected,
                              faces_uvs=self.ft[self.ANIMAL].repeat(data_batch_size, 1, 1),
                              verts_uvs=self.vt[self.ANIMAL].repeat(data_batch_size, 1, 1), sampling_mode="nearest").to(self.device)
        torch_mesh = Meshes(verts=verts, faces=self.faces.repeat(verts.shape[0], 1, 1), textures=textures).to(
            self.device)
        # visualize_3d_mesh(verts.reshape(-1, 3), self.faces)
        # visualize_texture_mesh(torch_mesh[0])
        
        return torch_mesh

    def get_pose_gt(self, pose_selected, cameraindex): 
        # rotate the model given the render axis and camera angle
        # rotate the model to y axis is height
        pose_origianl = pose_selected[:, :3]
        pose_origianl_matrix = axis_angle_to_matrix(pose_origianl)  
        
        if cameraindex == 0:
            # only sample one camera angle
            angle_random = (np.random.random(size = 1) *360).repeat(pose_selected.shape[0])
        else:
            # # sample pose_selected.shape[0] camera angles
            # print(pose_selected.shape)
            angle_random = np.random.random(size = pose_selected.shape[0]) *360
        # print(f"angle_random: {angle_random}")
        rot_candidate_4gt = axis_angle_to_matrix(torch.Tensor([[0, -np.radians(i), 0] for i in angle_random]))
        pose_updated_matrix = torch.matmul(torch.matmul(rot_candidate_4gt, self.rot_model_4gt),pose_origianl_matrix)  
        pose_update = rotmat_to_axis_angle(pose_updated_matrix.unsqueeze(1),number=1)  
        pose_selected[:, :3] = pose_update
        # print(f"pose_update: {pose_update}")
        #### pose update done #####################################
        return pose_selected

    def get_trans_gt(self, verts, shapeflag): 
        trans = torch.mean(verts, axis=1) 
        trans = -trans + torch.tensor([[0., 0., 0.]]).float().to(self.device)
        return trans.unsqueeze(1)

    def random_SMAL_pose_params(self, flag, data_batch_size, interval=8, exclude = False): 
        if self.FLAG == 'TEST':
            selected_pose_set = self.pose 
            selected_pose_label = self.pose_label 
        elif self.FLAG == 'TRAIN':
            selected_pose_set = self.pose_training 
            selected_pose_label = self.pose_training_label 
        else:
            selected_pose_set = self.pose_valid 
            selected_pose_label = self.pose_valid_label 
        data_len = int(selected_pose_set.shape[0])
        index = np.random.randint(low=0, high=int(data_len / interval), size=(data_batch_size))
        pose_selected = selected_pose_set[::interval, :][index, :]
        # print(f"pose_key: {self.pose_key[index]}")
        pose_class = selected_pose_label[::interval][index]
        return pose_selected, pose_class

    def random_SMAL_shape_params(self, flag, data_batch_size):
        if self.ANIMAL == "canidae" or self.ANIMAL == "equidae":
            shape_selected = torch.tensor(np.random.normal(0, 1, size=(data_batch_size, 9))).float()
            shape_class = np.array([0  for i in range(data_batch_size)])
        else:
            shape_selected = torch.tensor(np.random.normal(0, 0.2, size=(data_batch_size, 20))).float()
            # shape_selected = torch.zeros((data_batch_size, 20)).float()
            shape_class = np.array([0  for i in range(data_batch_size)])
        return shape_selected, shape_class

    def random_texture(self, flag, data_batch_size): 
        if self.FLAG == 'TEST':
            selected_texture_label = self.texture_testindex
            # raise NotImplementedError
        elif self.FLAG == 'TRAIN':
            selected_texture_label = self.texture_trainindex
        else:
            selected_texture_label = self.texture_validindex
        texture_key = np.random.choice(list(selected_texture_label.keys()))
        index = np.random.randint(low=0, high=len(selected_texture_label[texture_key]), size=(data_batch_size))
        texture_class = selected_texture_label[texture_key][index]
        # print(f"texture_key: {texture_key}, texture_class: {texture_class}")
        init_texture_tensor =self.texture[texture_key][texture_class]
        return init_texture_tensor, texture_class
    
    def random_background(self, flag, data_batch_size): 
        if self.FLAG == 'TEST':
            selected_background_set = self.background_train
            selected_background_index = self.background_trainingindex
        elif self.FLAG == 'TRAIN':
            selected_background_set = self.background_train
            selected_background_index = self.background_trainingindex
        else:
            selected_background_set = self.background_train
            selected_background_index = self.background_trainingindex

        index = np.random.randint(low=0, high=int(len(selected_background_set)), size=(data_batch_size))
        background_name = [selected_background_set[i] for i in index]
        return background_name
    
    def change_background(self,init_images_tensor, mask_image_tensor, background_name): 
        '''
        mask_image_tensor [B,1,256,256]; init_images_tensor[..., :3] [B,3,256,256]; background_name: [B]
        '''
        new_images = torch.zeros_like(init_images_tensor)
        # Iterate over each image in the batch
        for i in range(init_images_tensor.shape[0]):
            # Read the background image
            background_PIL = Image.open( background_name[i]).convert("RGB")
            background_tensor = self.background_transform(background_PIL)
            mask_expanded = mask_image_tensor[i].expand_as(init_images_tensor[i])
            # Use the mask to select foreground
            foreground = init_images_tensor[i] * mask_expanded
            # Use the inverted mask to select background
            background = background_tensor * (1 - mask_expanded)
            # Combine
            new_images[i] = foreground + background        
        return new_images

    def obtain_SMAL_pair_w_texture(self, label, data_batch_size, poseflag, shapeflag, textureflag, interval=8,cameraindex=0): 
        if label ==  1:
            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=data_batch_size,
                                                                    interval=interval) 
            pose_selected[1, :3] = pose_selected[0, :3]  # with the same root
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=1)
            shape_selected = shape_selected.repeat(data_batch_size, 1)
            shapeclass = shapeclass.repeat(data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=1)
            texture_selected = texture_selected.repeat(data_batch_size, 1, 1, 1)
            textureclass = textureclass.repeat(data_batch_size)
        elif label == 2:  # label to 2: appearance space: change appearance; only one SMAL one cam, but two texture
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=data_batch_size)

            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=1, interval=interval)
            pose_selected = pose_selected.repeat(data_batch_size, 1)
            poseclass = poseclass.repeat(data_batch_size)
        elif label == 3:  # label to 3: cam space: change cam ; only one SMAL one texture but two cam
            shape_selected, shapeclass = self.random_SMAL_shape_params(flag=shapeflag, data_batch_size=1)
            shape_selected = shape_selected.repeat(data_batch_size, 1)
            shapeclass = shapeclass.repeat(data_batch_size)
            pose_selected, poseclass = self.random_SMAL_pose_params(flag=poseflag, data_batch_size=1, interval=interval)
            pose_selected = pose_selected.repeat(data_batch_size, 1)
            poseclass = poseclass.repeat(data_batch_size)
            texture_selected, textureclass = self.random_texture(flag=textureflag, data_batch_size=1)
            texture_selected = texture_selected.repeat(data_batch_size, 1, 1, 1)
            textureclass = textureclass.repeat(data_batch_size)
        # save_pose_and_texture(pose_selected, f"../scripts/pose_selected_{self.ANIMAL}.npz")
        # print(f"pose_selected before update: {pose_selected[:, :3]}")
        pose_selected = self.get_pose_gt(pose_selected, cameraindex)
        # print(f"Label: {label}")
        # print(f"pose_selected after update: {pose_selected[:, :3]}")
        verts, _, _ = self.smal_model[self.ANIMAL](beta=shape_selected.to(self.device),
                                      theta=pose_selected.to(self.device),
                                      trans=torch.zeros((data_batch_size, 3)).to(self.device))
        # print(f"verts.shape: {verts.shape}")

        
        v = verts[0].reshape(-1, 3)
        # visualize_3d_mesh(v, self.faces)
        trans = self.get_trans_gt(verts, shapeflag)
        verts = verts + trans
        # print(f"verts.shape: {verts.shape} and faces.shape: {self.faces.repeat(verts.shape[0], 1, 1).shape}")
        torch_mesh = self.obtain_SMAL_meshes(data_batch_size, verts, texture_selected)
        return torch_mesh, shapeclass, poseclass, textureclass, shape_selected, pose_selected, trans
    
    def obtain_camera_label(self, kp_3d_tensor): 
        batch = kp_3d_tensor.shape[0]  # [N, 17, 3]
        cameraclass = []
        for i in range(batch):
            kp_3d_now = kp_3d_tensor[i, ...]
            if kp_3d_now[2, 0] < kp_3d_now[4, 0]:  # nose x < tail x
                left = 1;
                right = 0
            else:
                right = 1;
                left = 0
            if kp_3d_now[2, 2] > kp_3d_now[4, 2]:  # nose z> tail z
                away = 1;
                toward = 0
            else:
                toward = 1;
                away = 0
            cameraclass.append([toward, away, left, right])
        return np.array(cameraclass)
    
    def get_image(self, data_batch_size, poseflag=None, shapeflag=None, textureflag=None, interval=8, cameraindex=None): 
        verts, textures, shapeclass, poseclass, textureclass, shape_selected_gt, pose_selected_gt, trans_gt = [], [], [], [], [], [],[],[]
        for i in range(data_batch_size):
            if cameraindex is None:
                cameraindex = 1
            v, texture, shapecla, posecla, texturecla, shape_selected, pose_selected, trans = self.obtain_SMAL_vertices(1,  poseflag = None, shapeflag = None, textureflag = None, interval = interval,
                                                                                  cameraindex=cameraindex)
            verts.append(v)
            textures.append(texture)
            shapeclass.append(shapecla)
            poseclass.append(posecla)
            textureclass.append(texturecla)
            shape_selected_gt.append(shape_selected)
            pose_selected_gt.append(pose_selected)
            trans_gt.append(trans)
        verts = torch.cat(verts)
        textures = torch.cat(textures)
        shapeclass = np.concatenate(shapeclass)
        poseclass = np.concatenate(poseclass)
        textureclass = np.concatenate(textureclass)
        shape_selected_gt = torch.cat(shape_selected_gt)
        pose_selected_gt = torch.cat(pose_selected_gt)
        trans_gt = torch.cat(trans_gt)
        meshes = self.obtain_SMAL_meshes(data_batch_size=data_batch_size, verts=verts, texture_selected=textures)
        return meshes, shapeclass, poseclass, textureclass, [[0] * data_batch_size][0], shape_selected_gt, pose_selected_gt, trans_gt

    def get_image_pair_label(self, data_batch_size, poseflag=None, shapeflag=None, textureflag=None, interval=8,
                             cameraindex=None): 
        probablity = np.random.uniform(0, 1)
        # Determine the class
        if probablity < 1 / 3.:
            label = 1  # label to 1: pose space: change pose ;
        elif 1 / 3. <= probablity < 2 / 3.:
            label = 2  # label to 2: appearance space: change appearance
        else:
            label = 3  # label to 3: cam space: change cam
            
        if label == 1 or label == 2:
            cameraindex = 0 
        elif label == 3:
            cameraindex = 1

        
    
        meshes, shapeclass, poseclass, textureclass, shape_selected, pose_selected, trans = self.obtain_SMAL_pair_w_texture(label, data_batch_size, poseflag = None,
                                                                                      shapeflag= None, textureflag = None,
                                                                                      interval=self.num_texture,
                                                                                      cameraindex=cameraindex)
        
        return meshes, shapeclass, poseclass, textureclass, label, shape_selected, pose_selected, trans
    

    def __getitem__(self, index):
        self.cameras, self.renderer = self.setup_weak_render(device=self.device,
                                                                 image_size=self.args.imgsize)
        if self.args.data_batch_size == 2:
            meshes, shapeclass, poseclass, textureclass, label, shape_selected_gt, pose_selected_gt, trans_gt = self.get_image_pair_label(
                data_batch_size=self.args.data_batch_size,
                poseflag=None, shapeflag=None, textureflag=None,
                interval=self.args.useinterval)
        else:
            meshes, shapeclass, poseclass, textureclass, label, shape_selected_gt, pose_selected_gt, trans_gt = self.get_image(self.args.data_batch_size,
                                                                                poseflag=None, shapeflag=None,
                                                                                textureflag=None,
                                                                                interval=self.args.useinterval,
                                                                                cameraindex=None)  
            
        init_images_tensor, mask_image_tensor, kp_3d_tensor, kp_2d_tensor = render_image_mask_and_save_with_preset_render(
                renderer=self.renderer,
                cameras=self.cameras,
                mesh=meshes,
                image_size=self.args.imgsize,
                save_intermediate=True)                                                            

        init_images_tensor = init_images_tensor.detach().cpu()
        mask_image_tensor = mask_image_tensor.detach().cpu()
        kp_2d_tensor = kp_2d_tensor.detach().cpu()
        kp_3d_tensor = kp_3d_tensor.detach().cpu()

        shape_selected_gt = shape_selected_gt.detach().cpu()
        pose_selected_gt = pose_selected_gt.detach().cpu()
        trans_gt = trans_gt.detach().cpu()

        mask_image_tensor = mask_image_tensor.permute(0, 3, 1, 2)  
        temp_images_tensor = init_images_tensor[..., :3].permute(0, 3, 1, 2)  

        if self.args.background:
            background_name = self.random_background(flag = None, data_batch_size = self.args.data_batch_size)
            temp_images_tensor = self.change_background(temp_images_tensor, mask_image_tensor, background_name)

        # print(f"init_images_tensor.shape: {temp_images_tensor.shape}")
        image = temp_images_tensor.cpu().numpy()
        image = np.transpose(image, (0, 2, 3, 1))
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot first image
        ax1.imshow(image[0])
        ax1.axis('off')
        ax1.set_title('Image 1')
        
        # Plot second image
        ax2.imshow(image[1])
        ax2.axis('off')
        ax2.set_title('Image 2')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        save_plotted_image(image[0], f"../W3Z_AnimalClassifier/{self.ANIMAL}.jpg")

        if self.transform:
            init_images_tensor = self.process_image(temp_images_tensor)  

        shapeclass = torch.tensor(shapeclass.tolist(), dtype=int)
        poseclass = torch.tensor(poseclass.tolist(), dtype=int)
        textureclass = torch.tensor(textureclass.tolist(), dtype=int)
        camera_index_class = torch.tensor(
            self.obtain_camera_label(kp_3d_tensor))  # not used 
        label_tensor = torch.tensor([label], dtype=int)  

        out = (*map(none_to_nan, (
            init_images_tensor, mask_image_tensor, temp_images_tensor, shapeclass, poseclass, textureclass,
            camera_index_class, kp_3d_tensor, None, label_tensor, index, kp_2d_tensor, shape_selected_gt,pose_selected_gt,trans_gt)),)  # for batch collation

        del meshes
        # Clear up the memory
        torch.cuda.empty_cache()
        # Call the garbage collector
        gc.collect()
        return out

def save_texture(texture_selected, save_path):
    """
    Save the selected texture as an image.
    
    Args:
        texture_selected: The texture tensor from TexturesUV
        save_path: Path where to save the texture
    """
    # Convert tensor to numpy array and adjust format
    texture_np = texture_selected.cpu().numpy()
    
    # If texture is in range [0,1], convert to [0,255]
    if texture_np.max() <= 1.0:
        texture_np = (texture_np * 255).astype(np.uint8)
    
    # If texture has shape [1, H, W, 3], remove the batch dimension
    if texture_np.shape[0] == 1:
        texture_np = texture_np[0]
    
    # Ensure the texture is in RGB format
    if texture_np.shape[-1] != 3:
        texture_np = texture_np.transpose(1, 2, 0)
    
    # Save the image
    import cv2
    cv2.imwrite(save_path, cv2.cvtColor(texture_np, cv2.COLOR_RGB2BGR))

def visualize_texture_mesh(mesh, num_views=4, image_size=256):
    """
    Visualize a textured mesh from multiple angles.
    
    Args:
        mesh: PyTorch3D mesh with TexturesUV
        num_views: Number of views to render
        image_size: Size of rendered images
        distance: Camera distance from object
    """
    device = mesh.device
    verts = mesh.verts_packed()
    min_xyz = verts.min(dim=0).values
    max_xyz = verts.max(dim=0).values
    bbox_size = max_xyz - min_xyz
    center = (max_xyz + min_xyz) / 2.0
    # visualize_mesh_with_axes(mesh)

    object_radius = bbox_size.norm() / 2
    distance = object_radius * 2.5
    # Create figure
    fig = plt.figure(figsize=(15, 4))
    
    # Create cameras at different angles
    angles = torch.linspace(0, 360, num_views + 1)[:-1]
    
    for idx, angle in enumerate(angles):
        # Rotate around Z-axis
        # R = torch.tensor([
        #     [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],  # X
        #     [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],   # Y
        #     [0, 0, 1]                                                     # Z
        # ]).to(device)
        
        # # Position camera along X-axis (looking towards -X)
        # T = torch.tensor([distance * 2, 0, 0]).to(device) 
        
        R = torch.tensor([
            [np.cos(np.radians(angle)), 0, -np.sin(np.radians(angle))],
            [0, 1, 0],
            [np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ]).to(device)
        
        T = torch.tensor([0, 0, distance]).to(device) 
        
        cameras = FoVPerspectiveCameras(
            device=device,
            R=R[None],  # Add batch dimension: (1, 3, 3)
            T=T[None],  # Add batch dimension: (1, 3)
        )
        
        # Lighting
        lights = PointLights(
            device=device,
            location=[[0.0, 2.0, 2.0]],
            ambient_color=[[0.5, 0.5, 0.5]],
            diffuse_color=[[0.3, 0.3, 0.3]],
            specular_color=[[0.2, 0.2, 0.2]]
        )
        # lights = AmbientLights(device=device)
        
        # Renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=image_size,
                    blur_radius=0.0,
                    faces_per_pixel=1
                )
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

        # visualize_mesh_scene(mesh, cameras.get_camera_center(), distance=distance)
        # Render
        image = renderer(mesh)
        image = image[0, ..., :3].cpu().numpy()
        
        # Add to subplot
        plt.subplot(1, num_views, idx + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'View {idx + 1}')
    
    plt.tight_layout()
    plt.show()
    
def visualize_mesh_with_axes(mesh, axis_length=1.0):
    """
    Visualize mesh with coordinate axes
    
    Args:
        mesh: PyTorch3D mesh
        axis_length: Length of coordinate axes
    """
    device = mesh.device
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                    triangles=faces,
                    alpha=0.7,
                    color='gray')
    
    # Get mesh center for axis origin
    center = verts.mean(axis=0)
    
    # Plot coordinate axes
    # X-axis (red)
    ax.quiver(center[0], center[1], center[2], 
             axis_length, 0, 0, 
             color='red', 
             arrow_length_ratio=0.1)
    
    # Y-axis (green)
    ax.quiver(center[0], center[1], center[2], 
             0, axis_length, 0, 
             color='green', 
             arrow_length_ratio=0.1)
    
    # Z-axis (blue)
    ax.quiver(center[0], center[1], center[2], 
             0, 0, axis_length, 
             color='blue', 
             arrow_length_ratio=0.1)
    
    # Add axis labels
    ax.text(center[0] + axis_length, center[1], center[2], "X", color='red')
    ax.text(center[0], center[1] + axis_length, center[2], "Y", color='green')
    ax.text(center[0], center[1], center[2] + axis_length, "Z", color='blue')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Make axes equal
    max_range = np.array([
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_box_aspect([1,1,1])
    # Add title
    plt.title('Mesh with Coordinate Axes')
    
    # Add legend
    ax.legend(['Mesh', 'X-axis', 'Y-axis', 'Z-axis'])
    
    plt.show()

def visualize_mesh_scene(mesh, camera_pos, fov_degrees=60, light_pos=None, distance=2.0):
    """
    Visualize mesh with camera frustum, light position, and viewing direction
    """
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                    triangles=faces,
                    alpha=0.3,
                    color='gray')
    
    # Convert camera position to numpy and take first row
    camera_pos = camera_pos[0].cpu().numpy()
    
    # Calculate frustum vertices
    fov_rad = np.radians(fov_degrees)
    near = 0.1
    far = distance * 2
    
    # Calculate frustum dimensions at near and far planes
    near_height = 2 * np.tan(fov_rad / 2) * near
    near_width = near_height
    far_height = 2 * np.tan(fov_rad / 2) * far
    far_width = far_height
    
    # Define frustum vertices
    frustum_verts = np.array([
        # Near plane
        camera_pos + np.array([-near_width/2, -near_height/2, near]),
        camera_pos + np.array([near_width/2, -near_height/2, near]),
        camera_pos + np.array([near_width/2, near_height/2, near]),
        camera_pos + np.array([-near_width/2, near_height/2, near]),
        # Far plane
        camera_pos + np.array([-far_width/2, -far_height/2, far]),
        camera_pos + np.array([far_width/2, -far_height/2, far]),
        camera_pos + np.array([far_width/2, far_height/2, far]),
        camera_pos + np.array([-far_width/2, far_height/2, far])
    ])
    
    # Draw camera position
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], 
              color='blue', s=100, label='Camera')
    
    # Draw frustum lines
    frustum_edges = [
        # Near plane
        (0,1), (1,2), (2,3), (3,0),
        # Far plane
        (4,5), (5,6), (6,7), (7,4),
        # Connections
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    for start, end in frustum_edges:
        ax.plot3D([frustum_verts[start][0], frustum_verts[end][0]],
                 [frustum_verts[start][1], frustum_verts[end][1]],
                 [frustum_verts[start][2], frustum_verts[end][2]],
                 color='blue', alpha=0.5)
    
    # Draw light position if provided
    if light_pos is not None:
        light_pos = light_pos[0].cpu().numpy()  # Take first row of light_pos too
        ax.scatter(light_pos[0], light_pos[1], light_pos[2], 
                  color='yellow', s=100, label='Light')
        
        # Draw line from light to mesh center
        mesh_center = verts.mean(axis=0)
        light_dir = mesh_center - light_pos
        ax.quiver(light_pos[0], light_pos[1], light_pos[2],
                 light_dir[0], light_dir[1], light_dir[2],
                 color='yellow', 
                 alpha=0.5,
                 arrow_length_ratio=0.1)
    
    # Draw viewing direction
    view_dir = np.array([0, 0, distance])
    ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
             view_dir[0], view_dir[1], view_dir[2],
             color='blue', 
             alpha=0.7,
             arrow_length_ratio=0.1,
             label='View Direction')
    
    # Set equal aspects and limits
    max_range = np.array([
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min()
    ]).max() * 0.6
    
    mid_x = verts[:, 0].mean()
    mid_y = verts[:, 1].mean()
    mid_z = verts[:, 2].mean()
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.title('Mesh Scene with Camera Frustum and Light')
    plt.show()

def transform_vertices(verts):
    """
    Apply rotation transformations to vertices
    Args:
        verts: tensor of shape [batch_size, num_verts, 3]
    Returns:
        transformed vertices
    """
    # Convert to numpy for transformation
    verts_np = verts.cpu().numpy()
    
    # Create rotation matrices
    rotation_x = trimesh.transformations.rotation_matrix(
        angle=-np.radians(90),
        direction=[1, 0, 0],
        point=[0, 0, 0]
    )
    
    rotation_y = trimesh.transformations.rotation_matrix(
        angle=-np.radians(90),
        direction=[0, 1, 0],
        point=[0, 0, 0]
    )
    
    # Combine transformations
    transform = np.matmul(rotation_y, rotation_x)
    
    # Apply transformation to each batch
    transformed_verts = []
    for batch_verts in verts_np:
        # Add homogeneous coordinate (1) to each vertex
        homogeneous_verts = np.hstack([batch_verts, np.ones((batch_verts.shape[0], 1))])
        
        # Apply transformation
        transformed = np.dot(homogeneous_verts, transform.T)
        
        # Remove homogeneous coordinate
        transformed = transformed[:, :3]
        transformed_verts.append(transformed)
    
    # Stack batches and convert back to tensor
    transformed_verts = np.stack(transformed_verts)
    return torch.tensor(transformed_verts, device=verts.device, dtype=verts.dtype)

def save_pose_and_texture(pose_selected, save_dir, index=0):
    """
    Save pose and texture data to npz file
    
    Args:
        pose_selected: Selected pose tensor
        texture_selected: Selected texture tensor
        save_dir: Directory to save the npz file
        index: Index or identifier for the file name
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    pose_np = pose_selected.cpu().numpy()
    
    # Create file path
    save_path = os.path.join(save_dir, f'pose_texture_{index}.npz')
    
    # Save to npz file
    np.savez(save_path,
             pose=pose_np)
    
    print(f"Saved pose and texture to {save_path}")

def save_plotted_image(image, save_path, dpi=300):
    """
    Save the plotted image to file
    
    Args:
        image: Image array or tensor
        save_path: Path where to save the image
        dpi: Resolution for the saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Create figure with tight layout
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')  # Remove axes
    
    # Save with tight bounds and no padding
    plt.savefig(save_path, 
                bbox_inches='tight',
                pad_inches=0,
                dpi=dpi)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved image to {save_path}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--animal", type=str, default="hippopotamus")
    args.add_argument("--device", type=str, default="cuda")
    args.add_argument('--imgsize', type=int, default=256, help='number of workers')
    args.add_argument('--data_batch_size', type=int, default=2 , help='batch size; before is 36')
    args.add_argument('--useinterval', type=int, default=8, help='number of interval of the data')
    args.add_argument('--background', type=bool, default=False, help='background')









    args = args.parse_args()
    for i in ["equidae", "felidae",  "bovidae", "hippopotamus", "canidae" ]:
        pipeline = AnimalPipe(args, args.device, 100, FLAG='TRAIN', ANIMAL=i)
        sample = pipeline[0]  # or any index
        # sample1 = sample[0]
        # image1 = sample1[0]
        # print(f"image1.shape: {image1.shape}")  # shape: (256, 256, 4)
        # image1 = transforms.ToPILImage()(image1)
        # # image = image1[0, ..., :3].cpu().numpy()
    
        # # image2 = sample1[1]
        # # print(sample1.shape)
        # # print(image1.shape)
        # # print(image2.shape)
        # # Convert from torch to numpy if needed
        # # if isinstance(image1, torch.Tensor):
        # #     image1 = image1.permute(1, 2, 0).numpy()
        #     # image2 = image2.permute(1, 2, 0).numpy()

        # # Plot both
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].imshow(image1)
        # axs[0].set_title("Image 1")
        # axs[0].axis('off')

        # # axs[1].imshow(image2)
        # # axs[1].set_title("Image 2")
        # # axs[1].axis('off')

        # plt.tight_layout()
        # plt.show()