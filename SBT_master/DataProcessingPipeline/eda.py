import pickle
import cv2

# Load the saved data
with open('/home/harshr/bags/BL_2024-09-04/BL_2024-09-04_19-09-17_chunk0000.pkl', 'rb') as f:
    data = pickle.load(f)

# Iterate through the data
for i in range(len(data['image_paths']['thermal'])):
    # Access image paths
    # thermal_patch_path = data['image_paths']['thermal'][i]
    # left_rgb_patch_path = data['image_paths']['left_rgb'][i]
    # right_rgb_patch_path = data['image_paths']['right_rgb'][i]
    # depth_patch_path = data['image_paths']['depth'][i]
    
    # # Load images
    # thermal_patch = cv2.imread(thermal_patch_path)
    # left_rgb_patch = cv2.imread(left_rgb_patch_path)
    # right_rgb_patch = cv2.imread(right_rgb_patch_path)
    # depth_patch = cv2.imread(depth_patch_path, cv2.IMREAD_GRAYSCALE)  # Depth is single-channel
    
    # Access IMU data
    roll = data['roll_pitch_yaw'][i][0]
    pitch = data['roll_pitch_yaw'][i][1]
    yaw = data['roll_pitch_yaw'][i][2]
    
    # Now, you have:
    # - thermal_patch: Thermal image at index i
    # - left_rgb_patch: Left RGB image at index i
    # - right_rgb_patch: Right RGB image at index i
    # - depth_patch: Depth image at index i
    # - roll, pitch, yaw: IMU data at index i
    
    # Example usage: Display images and print IMU data
    # cv2.imshow('Thermal Image', thermal_patch)
    # cv2.imshow('Left RGB Image', left_rgb_patch)
    # cv2.imshow('Right RGB Image', right_rgb_patch)
    # cv2.imshow('Depth Image', depth_patch)
    print(f"IMU Data at Index {i}: Roll={roll}, Pitch={pitch}, Yaw={yaw}")
    cv2.waitKey(1)  # Adjust as needed for your application

cv2.destroyAllWindows()
