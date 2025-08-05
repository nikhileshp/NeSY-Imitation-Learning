import cv2
import os
import sys
from models.OC_Atari.ocatari.core import OCAtari
from models.OC_Atari.ocatari.vision.utils import find_objects, mark_bb, make_darker, match_objects
from models.OC_Atari.ocatari.vision.game_objects import GameObject, NoObject
# Set the directory containing your images

# objects_colors = {"player": [[187, 187, 53], [236, 236, 236]], "diver": [66, 72, 200], "background_water": [0, 28, 136],
objects_colors = {"player": [187, 187, 53], "diver": [66, 72, 200], "background_water": [0, 28, 136],
                  "player_score": [210, 210, 64], "oxygen_bar": [214, 214, 214], "lives": [210, 210, 64],
                  "logo": [66, 72, 200], "player_missile": [187, 187, 53], "oxygen_bar_depleted": [163, 57, 21],
                  "oxygen_logo": [0, 0, 0], "collected_diver": [24, 26, 167], "enemy_missile": [66, 72, 200],
                  "submarine": [170, 170, 170]}

enemy_colors = {"green": [92, 186, 92], "orange": [198, 108, 58], "yellow": [160, 171, 79], "lightgreen": [72, 160, 72],
                "pink": [198, 89, 179]}

class PlayerMissile(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 187, 187, 53
    

class Diver(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 66, 72, 200


class EnemyMissile(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 66, 72, 200



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python convert_to_video.py image_folder output_video [fps]')
        exit(1)
    image_folder = sys.argv[1]
    output_video = "test_output.mp4" if len(sys.argv) < 3 else sys.argv[2]
    fps =   30  # Default frames per second
    if len(sys.argv) == 4:
        fps = int(sys.argv[3])


    # Get all image files, sorted by name
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    
    height, width, _ = first_frame.shape
    


  
    img_array = cv2.imread(os.path.join(image_folder, images[120]))
    obs = img_array.copy()
    player = []
    for color in objects_colors["player"]:
        player.extend(find_objects(obs, color, closing_dist=8))

    print("Found player objects:", player)

    divers_and_missiles = find_objects(
        obs, objects_colors["diver"], closing_dist=1)
    divers = []
    missiles = []
    for dm in divers_and_missiles:
        if dm[1] < 190 and dm[2] > 2 and dm[3] > 5:
            divers.append(dm)
        elif dm[1] < 190 and dm[2] > 2:
            missiles.append(dm)



    print("Found divers:", divers)
    print("Found missiles:", missiles)    

    shark = []
    for enemyColor in enemy_colors.values():
        shark.extend(find_objects(obs, enemyColor, min_distance=1))

    print("Found sharks:", shark)

    submarine = find_objects(obs, objects_colors["submarine"], min_distance=1)
    print("Found submarines:", submarine)

    oxygen_bar = find_objects(
        obs, objects_colors["oxygen_bar"], min_distance=1)
    print("Found oxygen bars:", oxygen_bar)


    # for obj in objects_colors:
        
    #     x = find_objects(img_array,color=objects_colors[obj],min_distance=1, maxx=width, maxy=height )
    #     if len(x) > 0:
    #         print("Colors for", obj, ":", objects_colors[obj])
    #         print(f"Found {obj}: {x} objects in {images[0]}")
    #     quit()
    # Read first image to get width and height
    

    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID' or 'avc1'
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each frame
    for img in images:
        frame = cv2.imread(os.path.join(image_folder, img))
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video}")