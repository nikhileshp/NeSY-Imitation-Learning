import cv2
import os
import sys
from models.OC_Atari.ocatari.core import OCAtari
from models.OC_Atari.ocatari.vision.utils import find_objects, mark_bb, make_darker, match_objects
from models.OC_Atari.ocatari.vision.game_objects import GameObject, NoObject
# Set the directory containing your images

# objects_colors = {"player": [[187, 187, 53], [236, 236, 236]], "diver": [66, 72, 200], "background_water": [0, 28, 136],
objects_colors = {"player": [[187, 187, 53], [236, 236, 236]], "diver": [66, 72, 200], "background_water": [0, 28, 136],
                  "player_score": [210, 210, 64], "oxygen_bar": [214, 214, 214], "lives": [210, 210, 64],
                  "logo": [66, 72, 200], "player_missile": [187, 187, 53], "oxygen_bar_depleted": [163, 57, 21],
                  "oxygen_logo": [0, 0, 0], "collected_diver": [24, 26, 167], "enemy_missile": [66, 72, 200],
                  "submarine": [170, 170, 170]}

enemy_colors = {"green": [92, 186, 92], "orange": [198, 108, 58], "yellow": [160, 171, 79], "lightgreen": [72, 160, 72],
                "pink": [198, 89, 179]}

class Player(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 187, 187, 53


class Diver(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 66, 72, 200


class Shark(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 92, 186, 92


class Submarine(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 170, 170, 170


class PlayerMissile(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 187, 187, 53


class OxygenBar(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 214, 214, 214


class OxygenBarDepleted(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 163, 57, 21


class OxygenBarLogo(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 0, 0, 0


class PlayerScore(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 210, 210, 64


class Lives(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 210, 210, 64


class CollectedDiver(GameObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rgb = 24, 26, 167


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
    fps =   1  # Default frames per second
    if len(sys.argv) == 4:
        fps = int(sys.argv[3])


    # Get all image files, sorted by the index after the last underscore
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    
    height, width, _ = first_frame.shape
    

    # Loop through images and process every 30th image
    for i, img_name in enumerate(images):
        if i % fps != 0:
            continue
        img_array = cv2.imread(os.path.join(image_folder, img_name))
        obs = img_array.copy()
        player = []
        for color in objects_colors["player"]:
                player.extend(find_objects(obs, color, min_distance=1))


        # Print the image number and found objects
        print(f"Processing {img_name}:")
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

        submarine = find_objects(obs, objects_colors["submarine"], min_distance=1)

        
        print("Found submarines:", submarine)

        print("Found divers:", divers)
        print("Found missiles:", missiles)    

        lives = []
        for enemyColor in enemy_colors.values():
            lives.extend(find_objects(obs, enemyColor, min_distance=1))

        print("Found lives:", lives)

        submarine = find_objects(obs, objects_colors["submarine"], min_distance=1)
        print("Found submarines:", submarine)

        # Unable to find any oxygen bar, it should be in a red rectangle


        oxygen_bar = find_objects(
            obs, objects_colors["oxygen_bar"], min_distance=1)
        print("Found oxygen bars:", oxygen_bar)
        lives = find_objects(obs, objects_colors["lives"], min_distance=1)
        
        score = find_objects(obs, objects_colors["player_score"], min_distance=1)
        print("Found score:", score)
        logo = find_objects(obs, objects_colors["logo"], min_distance=1)
        print("Found logos:", logo)
        oxygen_depleted = find_objects(
            obs, objects_colors["oxygen_bar_depleted"], min_distance=1)
        print("Found depleted oxygen bars:", oxygen_depleted)
        oxygen_logo = find_objects(
            obs, objects_colors["oxygen_logo"], min_distance=1)
    
        coll_diver = find_objects(obs, objects_colors["collected_diver"])
        print("Found collected divers:", coll_diver)


        # Mark found objects on the image
        for obj_list, color in [(player, (0, 255, 0)),
                                (divers, (255, 0, 0)),
                                (missiles, (0, 0, 255)),
                                (lives, (255, 255, 0)),
                                (submarine, (0, 255, 255)),
                                (oxygen_bar, (255, 0, 255)),
                                (lives, (0, 255, 0)),
                                (score, (0, 200, 200)),
                                (logo, (200, 0, 200)),
                                (oxygen_depleted, (100, 100, 100)),
                                (oxygen_logo, (50, 50, 50)),
                                (coll_diver, (150, 0, 150))]:

            for obj in obj_list:
                # Create a bounding box around the object
                mark_bb(obs, obj, color=color)
                
        
  

        cv2.imshow('Frame', obs)
        key = cv2.waitKey(0)  # Wait for a key press to move
        if key == 27:
            break
    cv2.destroyAllWindows()
        

    # for obj in objects_colors:
        
    #     x = find_objects(img_array,color=objects_colors[obj],min_distance=1, maxx=width, maxy=height )
    #     if len(x) > 0:
    #         print("Colors for", obj, ":", objects_colors[obj])
    #         print(f"Found {obj}: {x} objects in {images[0]}")
    #     quit()
    # Read first image to get width and height
    

    # Define video codec and writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID' or 'avc1'
    # video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # # Write each frame
    # for img in images:
    #     frame = cv2.imread(os.path.join(image_folder, img))
    #     video.write(frame)

    # video.release()
    # print(f"Video saved as {output_video}")