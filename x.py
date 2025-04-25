import numpy as np
import matplotlib.pyplot as plt

def visualize_3d_vertices(vertices, title="3D Vertices"):
    """
    Visualize 3D vertices in a scatter plot with equal axis scales.
    Args:
        vertices: numpy array of shape (N, 3) containing x, y, z coordinates
        title: title for the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more interactive
    ax.view_init(elev=30, azim=45)
    
    # Get the limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Find the greatest range value
    max_range = max(x_range, y_range, z_range)
    
    # Calculate the mid-points for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    
    # Set new limits based on the maximum range
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.show()

def visualize_3d_mesh(vertices, faces, title="3D Mesh"):
    """
    Visualize 3D mesh using vertices and faces.
    Args:
        vertices: numpy array of shape (N, 3) containing x, y, z coordinates
        faces: numpy array of shape (M, 3) containing vertex indices for triangles
        title: title for the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh using triangles
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    triangles=faces,
                    color='lightgray',
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more interactive
    ax.view_init(elev=30, azim=45)
    
    # Get the limits for all axes
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    # Calculate the range for each axis
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    
    # Find the greatest range value
    max_range = max(x_range, y_range, z_range)
    
    # Calculate the mid-points for each axis
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    
    # Set new limits based on the maximum range
    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.show()