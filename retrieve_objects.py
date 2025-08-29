import cv2
import os
import sys
import pandas as pd 
from models.OC_Atari.ocatari.core import OCAtari
from models.OC_Atari.ocatari.vision.utils import find_objects, mark_bb, make_darker, match_objects
from models.OC_Atari.ocatari.vision.game_objects import GameObject, NoObject
# Set the directory containing your images

# objects_colors = {"player": [[187, 187, 53], [236, 236, 236]], "diver": [66, 72, 200], "background_water": [0, 28, 136],


enemy_colors = {"green": [92, 186, 92], "orange": [[58, 108, 198], [111, 111, 111]],  "yellow": [[111,111,111],[79,171,160]], "lightgreen": [72, 160, 72],
                "pink": [[179,89,198],[192,70,146]]}


def above_water(obj):
    return obj[1] < 47

def right_of(obj1, obj2):
    """Check if obj1 is to the left of obj2."""
    return obj2[0] + obj2[2] <= obj1[0]

def left_of(obj1, obj2):
    """Check if obj1 is to the right of obj2."""
    return obj1[0] + obj1[2] <= obj2[0]


# Add three functions above_of, below_of and same_level_of to check if obj1 is above, below or at the same level as obj2
# Obj1 is on the same level of obj2 if the height of the body of obj2 is mostly overlapping the hight of the body of obj1
def above_of(obj1, obj2, min_coverage=0.51):
    """Check if obj1 is above obj2."""
    mid_y1 = obj1[1] + obj1[3] // 2
    coverage = max(0, min(obj1[1] + obj1[3], obj2[1] + obj2[3]) - max(obj1[1], obj2[1]))
    coverage_ratio = coverage / obj2[3]
    return mid_y1 < obj2[1] and (coverage_ratio < min_coverage)

def below_of(obj1, obj2, min_coverage=0.51):
    """Check if obj1 is below obj2."""
    mid_y1 = obj1[1] + obj1[3] // 2
    coverage = max(0, min(obj1[1] + obj1[3], obj2[1] + obj2[3]) - max(obj1[1], obj2[1]))
    coverage_ratio = coverage / obj2[3]
    return mid_y1 > (obj2[1] + obj2[3]) and (coverage_ratio < min_coverage)
    
def same_level_of(obj1, obj2, min_coverage=0.51):
    """Check if obj1 is at the same level as obj2."""
    mid_y1 = obj1[1] + obj1[3] // 2
    coverage = max(0, min(obj1[1] + obj1[3], obj2[1] + obj2[3]) - max(obj1[1], obj2[1]))
    coverage_ratio = coverage / obj2[3]
    return (obj2[1] <= mid_y1 <= (obj2[1] + obj2[3])) and (coverage_ratio >= min_coverage)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python convert_to_video.py image_folder output_video [fps]')
        exit(1)
    image_folder = sys.argv[1]

    # Text file path is one directory before image path
    text_file_path = os.path.join(image_folder + ".txt")
    if not os.path.exists(text_file_path):
        print(f"Text file {text_file_path} does not exist.")
        exit(1)

    print(text_file_path)

    output_video = "test_output.mp4" if len(sys.argv) < 3 else sys.argv[2]
    fps =  1  # Default frames per second
    if len(sys.argv) == 4:
        fps = int(sys.argv[3])



    # Get all image files, sorted by the index after the last underscore
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))],
                    key=lambda x: int(x.split('_')[-1].split('.')[0]))
    first_frame = cv2.imread(os.path.join(image_folder, images[0]))
    
    height, width, _ = first_frame.shape

    objects_colors = {"player": [[53, 187, 187]], "diver": [[200, 72, 66],[184, 50, 45]], "background_water": [[0, 28, 136]],
                  "player_score": [[210, 210, 64]], "oxygen_bar": [[214, 214, 214]], "lives": [[64, 210, 210]],
                  "logo": [[66, 72, 200]], "player_missile": [[142, 142, 142], [53, 187, 187]], "oxygen_bar_depleted": [[ 21, 57, 163]],
                  "oxygen_logo": [[0, 0, 0]], "collected_diver": [[ 167, 26, 24]], "enemy_missile": [[200, 72, 66]],
                  "submarine": [[170, 170, 170]]}


    # The gaze data text file has 7 columns: frameid, episode_id, score, duration, unclipped_reward, action and gaze_positions which are multiple x,y coordinates separated by commas. How do i read this information

    # Read the gaze information from the text file
    if not os.path.exists(text_file_path):
        print(f"Text file {text_file_path} does not exist.")
        exit(1)
    # Read the gaze information into a pandas DataFrame
    # Assuming the text file is comma-separated and has a header 
    # Read the 7th column as a list of tuples (x, y) coordinates

    with open(text_file_path, 'r') as f:
        gaze_data = f.readlines()
    gaze_info = []
    for line in gaze_data:
        parts = line.strip().split(',')
        # print(parts)
        if len(parts) <= 7:
            continue
        frameid = parts[0]
        episode_id = parts[1]
        score = parts[2]
        duration = float(parts[3])
        unclipped_reward = float(parts[4])
        action = int(parts[5])
        gaze_positions = parts[6:]
        # Convert gaze positions to a list of tuples (x, y)
        if len(gaze_positions) % 2 != 0:
            print(f"Warning: Odd number of gaze positions in line: {line.strip()}")
            continue
        # Convert gaze positions to integers and group them as (x, y) pairs
        gaze_positions = [float(pos) for pos in gaze_positions]
        gaze_positions = [(int(gaze_positions[i]), int(gaze_positions[i+1])) for i in range(0, len(gaze_positions), 2)]
        gaze_info.append({
            'frameid': frameid,
            'episode_id': episode_id,
            'score': score,
            'duration': duration,
            'unclipped_reward': unclipped_reward,
            'action': action,
            'gaze_positions': gaze_positions,
            "objects": "",
            'relationships': ""
        })
    gaze_df = pd.DataFrame(gaze_info)


    # Loop through images and process every 30th image
    for i, img_name in enumerate(images):
        if i % fps != 0:
            continue
        img_array = cv2.imread(os.path.join(image_folder, img_name))
        obs = img_array.copy()
        #Unable to find player reliably, so we look for both colors and combine results
        #Still unable to find in some frames, need better method

        enemy = []
        orange_yellow = enemy_colors["orange"] + enemy_colors["yellow"]
        enemy.extend(find_objects(obs, enemy_colors["green"], maxy=150, miny=40))
        enemy.extend(find_objects(obs, enemy_colors["lightgreen"], maxy=150, miny=40))
        enemy.extend(find_objects(obs, enemy_colors["pink"], maxy=150, miny=40))    
        enemy.extend(find_objects(obs, orange_yellow, maxy=150, miny=40))

        # Print the image number and found objects
        print(f"Processing {img_name}:")
        
        player = find_objects(obs, objects_colors["player"], size=(15,10), tol_s=(14,8))





        divers = find_objects(obs, objects_colors["diver"], size= (12,8), tol_s=6, miny=40, maxy=150)


        # print("Found enemy objects:", enemy)
        missiles = find_objects(obs, objects_colors["player_missile"], miny=40, maxy=150, size=(12,1), tol_s=(5,1))
        enemy_missiles = find_objects(obs, objects_colors["enemy_missile"], size=(8,4), tol_s=1, minx=20, miny=90, maxy=310)
        
        obj_remove_list = []
        for obj in enemy_missiles:
            for diver in divers:
                if obj[0] >= diver[0] and obj[0] <= (diver[0] + diver[2]) and obj[1] >= diver[1] and obj[1] <= (diver[1] + diver[3]):
                    # If the enemy missile is in the bounding box of the diver, then remove it
                    obj_remove_list.append(obj) 
        
        for diver in divers:
            if (diver[2] >= 6 and diver[2] <= 8) and diver[3] == 4:
                    enemy_missiles.append(diver)

        for obj in obj_remove_list:
            if obj in enemy_missiles:
                enemy_missiles.remove(obj) 

        
        for obj in missiles:
            for p in player:
                if obj[0] >= p[0] and obj[0] <= p[0] + p[2] and obj[1] >= p[1] and obj[1] <= p[1] + p[3]:
                    # If the missile is in the bounding box of the player, then remove it
                    missiles.remove(obj)
                    break
        # print("Found divers:", divers)
        # print("Found missiles:", missiles)    

        lives = find_objects(obs, objects_colors["lives"], closing_active=False, miny=20)
        
        # print("Found lives:", lives)

        enemy_submarine = find_objects(obs, objects_colors["submarine"], size=(8,10), tol_s=4, miny=56, maxy=150, min_distance=1)
        submarine_on_water = find_objects(obs, objects_colors["submarine"], miny=40, maxy=55, min_distance=1)

        enemy_submarine.extend(submarine_on_water)
        # print("Found enemy submarines:", enemy_submarine)
        # Unable to find any oxygen bar, it should be in a red rectangle
        oxygen_bar = find_objects(
            obs, objects_colors["oxygen_bar"], min_distance=1)
        # print("Found oxygen bars:", oxygen_bar)
        
        # score = find_objects(obs, objects_colors["player_score"], min_distance=1)
        # print("Found score:", score)
        # logo = find_objects(obs, objects_colors["logo"], min_distance=1)
        # print("Found logos:", logo)
        oxygen_depleted = find_objects(
            obs, objects_colors["oxygen_bar_depleted"], min_distance=1)
        # print("Found depleted oxygen bars:", oxygen_depleted)
        oxygen_logo = find_objects(
            obs, objects_colors["oxygen_logo"], min_distance=1)
    
        coll_diver = find_objects(obs, objects_colors["collected_diver"],closing_active=False)       
        # print("Found collected divers:", coll_diver)

        players_dict = {}
        divers_dict = {}
        coll_diver_dict = {}
        missiles_dict = {}
        lives_dict = {}
        oxygen_bar_dict = {}
        oxygen_depleted_dict = {}
        
        enemy_dict = {}
        objects_list = []
        
        for i, obj in enumerate(player):
            objects_list.append(f"player_{i}")
            if players_dict:
                players_dict[f"player_{i}"] = obj
            else:
                players_dict = {f"player_{i}": obj}

        for i, obj in enumerate(divers):
            objects_list.append(f"diver_{i}")
            if divers_dict:
                divers_dict[f"diver_{i}"] = obj
            else:
                divers_dict = {f"diver_{i}": obj}

        for i, obj in enumerate(coll_diver):
            objects_list.append(f"coll_diver_{i}")
            if coll_diver_dict:
                coll_diver_dict[f"coll_diver_{i}"] = obj
            else:
                coll_diver_dict = {f"coll_diver_{i}": obj}
        
        for i, obj in enumerate(missiles):
            objects_list.append(f"missile_{i}")
            if missiles_dict:
                missiles_dict[f"missile_{i}"] = obj
            else:
                missiles_dict = {f"missile_{i}": obj}

        for i, obj in enumerate(enemy_missiles):
            objects_list.append(f"enemy_missile_{i}")
            if enemy_dict:
                enemy_dict[f"enemy_missile_{i}"] = obj
            else:
                enemy_dict = {f"enemy_missile_{i}": obj}

        for i, obj in enumerate(lives):
            objects_list.append(f"life_{i}")
            if lives_dict:
                lives_dict[f"life_{i}"] = obj
            else:
                lives_dict = {f"life_{i}": obj}

        for i, obj in enumerate(oxygen_bar):
            objects_list.append(f"oxygen_bar_{i}")
            if oxygen_bar_dict:
                oxygen_bar_dict[f"oxygen_bar_{i}"] = obj
            else:
                oxygen_bar_dict = {f"oxygen_bar_{i}": obj}

        for i, obj in enumerate(oxygen_depleted):
            objects_list.append(f"oxygen_depleted_{i}")
            if oxygen_depleted_dict:
                oxygen_depleted_dict[f"oxygen_depleted_{i}"] = obj
            else:
                oxygen_depleted_dict = {f"oxygen_depleted_{i}": obj}

        for i, obj in enumerate(enemy_submarine):
            objects_list.append(f"enemy_submarine_{i}")
            if enemy_dict:
                enemy_dict[f"enemy_submarine_{i}"] = obj
            else:
                enemy_dict = {f"enemy_submarine_{i}": obj}

        for i, obj in enumerate(enemy):
            objects_list.append(f"enemy_{i}")
            if enemy_dict:
                enemy_dict[f"enemy_{i}"] = obj
            else:
                enemy_dict = {f"enemy_{i}": obj}

        print("Found objects:")
        if not player:
            pass
        else:
            print("Player objects:", players_dict)
        
        if not divers:
            pass
        else:
            print("Diver objects:", divers_dict)

        if not coll_diver:
            pass
        else:
            print("Collected Diver objects:", coll_diver_dict)

        if not missiles:
            pass
        else:
            print("Missile objects:", missiles_dict)

        if not enemy_missiles:
            pass
        else:
            print("Enemy Missile objects:", enemy_dict)
        if not lives:
            pass
        else:
            print("Lives objects:", lives_dict)

        if not oxygen_bar:
            pass
        else:
            print("Oxygen Bar objects:", oxygen_bar_dict)


        if not oxygen_depleted:
            pass
        else:
            print("Oxygen Depleted objects:", oxygen_depleted_dict)


        
        if not enemy:
            pass
        else:
            print("Enemy objects:", enemy_dict)
 
        
        print()
        print("Relationship between objects:")
        relationships = []
        try:
            # Check if player is above water
            if player and above_water(player[0]):
                print("aboveWater(player).")
                relationships.append((player[0], "water_surface", "aboveWater", "player"))
            else:
                print("belowWater(player).")
                relationships.append((player[0], "water_surface", "belowWater", "player"))
            # Check if player is to the left of any diver
            # Add relationships to a list with tuple information in the form of (obj1coord, obj2coord, relationship)
            
        
       
            for diver, coords in divers_dict.items():
            
                if left_of(player[0], coords):
                    relationships.append((player[0], coords, "leftOf", "player", diver))
                    print("leftOfDiver(player,", diver, ").")

                elif right_of(player[0], coords):
                    relationships.append((player[0], coords, "rightOf", "player", diver))
                    print("rightOfDiver(player,", diver, ").")

                if above_of(player[0], coords):
                    relationships.append((player[0], coords, "aboveOf", "player", diver))
                    print("aboveOfDiver(player,", diver, ").")

                elif below_of(player[0], coords):
                    relationships.append((player[0], coords, "belowOf", "player", diver))
                    print("belowOfDiver(player,", diver, ").")
                
                else:
                    relationships.append((player[0], coords, "sameLevelAs", "player", diver))
                    print("sameLevelAsDiver(player,", diver, ").")
            
        

            for enemy_obj, coords in enemy_dict.items():
                if left_of(player[0], coords):
                    relationships.append((player[0], coords, "leftOf", "player", enemy_obj))
                    print("leftOfEnemy(player,", enemy_obj, ").")

                elif right_of(player[0], coords):
                    relationships.append((player[0], coords, "rightOf", "player", enemy_obj))
                    print("rightOfEnemy(player,", enemy_obj, ").")

                if above_of(player[0], coords):
                    relationships.append((player[0], coords, "aboveOf", "player", enemy_obj))
                    print("aboveOfEnemy(player,", enemy_obj, ").")

                elif below_of(player[0], coords):
                    relationships.append((player[0], coords, "belowOf", "player", enemy_obj))
                    print("belowOfEnemy(player,", enemy_obj, ").")
                
                else:
                    relationships.append((player[0], coords, "sameLevelAs", "player", enemy_obj))
                    print("sameLevelAsEnemy(player,", enemy_obj, ").")

        except Exception as e:
            print("Error in finding relationships:", e)
            relationships = []
                # Save new dataframe to a new text file in the same directory as the original text file with _with_relationships appended to the filename
            new_text_file_path = text_file_path.replace(".txt", "_with_relationships.txt")
            # Remove gaze_positions column from the dataframe before saving
            gaze_df = gaze_df.drop(columns=['gaze_positions'])
            gaze_df.to_csv(new_text_file_path, index=False)

            print(f"Saved updated gaze data with relationships to {new_text_file_path}")
             # # Show a higher resolution window
            obs = cv2.resize(obs, (width*2, height*2), interpolation=cv2.INTER_NEAREST)
        
             # Run all frames in a loop without expecting a key press
            cv2.imshow('Frame', obs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            quit()
                
        # Using the relationship list, make a connection list which lists the connection between objects in the form of (objcoord1, objcoord2, [list of relationships])
        # Add the distance between the two objects in the connection list as well
        connection_list = []
        for rel in relationships:
            found = False
            for conn in connection_list:
                if rel[0] == conn[0] and rel[1] == conn[1]:
                    conn[2].append(rel[2])
                    
                    found = True
                    break
            if not found:
                if "water" in rel[1]:
                    continue
                distance = ((rel[0][0] - rel[1][0]) ** 2 + (rel[0][1] - rel[1][1]) ** 2) ** 0.5
                connection_list.append([rel[0], rel[1], [rel[2]], distance])
        print("Connection list:", connection_list)
        
        # Mark found objects on the image
        for obj_list, color in [(player, (0, 255, 0)),
                                (divers, (255, 0, 0)),
                                (missiles, (0, 255, 0)),
                                (lives, (255, 255, 0)),
                                (oxygen_bar, (255, 0, 255)),
                                (lives, (0, 255, 0)),
                                (enemy_submarine, (0, 0, 255)),
                                # (score, (0, 200, 200)),
                                # (logo, (200, 0, 200)),
                                (enemy_missiles, (0, 0, 255)),
                                (enemy, (0,0,255)),
                                (oxygen_depleted, (100, 100, 100)),
                                # (oxygen_logo, (50, 50, 50)),
                                (coll_diver, (150, 0, 150))]:

            for obj in obj_list:
                # Create a bounding box around the object
                mark_bb(obs, obj, color=color)
                # pass

        # For every connection in the connection list, draw a black line between the center of the two objects
        # Add text in the middle of the line with the relationships in format (capitalized first letter of each relationship separated by hyphen)
        for conn in connection_list:
            obj1, obj2 = conn[0], conn[1]
            center1 = (obj1[0] + obj1[2] // 2, obj1[1] + obj1[3] // 2)
            center2 = (obj2[0] + obj2[2] // 2, obj2[1] + obj2[3] // 2)
            cv2.line(obs, center1, center2, (0, 0, 0), 1)
            mid_point = ((center1[0] + center2[0]) // 2, (center1[1] + center2[1]) // 2)
            relationships_text = '-'.join([rel[0].upper()[0] for rel in conn[2]])
            cv2.putText(obs, relationships_text, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)


        # Mark the gaze positions on the image
        if not gaze_df.empty:
            #Match the frame id from the image name
            # Assuming the image name format is like 'frame_123.png'
            # Extract the frame id from the image name by removing .png
            frame_id = img_name.strip(".png")
            # print("Frame ID:", frame_id)
            
            # Add the relationships to the dataframe
            for obj in objects_list:
                if gaze_df.loc[gaze_df['frameid'] == frame_id, 'objects'].empty:
                    gaze_df.loc[gaze_df['frameid'] == frame_id, 'objects'] = f"{obj} , "
                else:
                    gaze_df.loc[gaze_df['frameid'] == frame_id, 'objects'] += f"{obj} , "
            for rel in relationships:
                if gaze_df.loc[gaze_df['frameid'] == frame_id, 'relationships'].empty:
                    if len(rel) ==4:
                        gaze_df.loc[gaze_df['frameid'] == frame_id, 'relationships'] = f"{rel[2]}({rel[3]}) , "
                    elif len(rel) ==5:
                        gaze_df.loc[gaze_df['frameid'] == frame_id, 'relationships'] = f"{rel[2]}({rel[3]},{rel[4]}) , "
                else:
                    if len(rel) ==4:
                        gaze_df.loc[gaze_df['frameid'] == frame_id, 'relationships'] += f"{rel[2]}({rel[3]}) , "
                    elif len(rel) ==5:
                        gaze_df.loc[gaze_df['frameid'] == frame_id, 'relationships'] += f"{rel[2]}({rel[3]},{rel[4]}) , "
            gaze_positions = gaze_df[gaze_df['frameid'] == frame_id]['gaze_positions'].values
            if len(gaze_positions) > 0:
                # Mark the tuples of gaze positions on the image
                for position in gaze_positions:
                    #Mark the position (x,y) with a red dot on the image
                    for x, y in position:
                        if 0 <= x < width and 0 <= y < height:
                            # Make a small red dot at the gaze position
                            
                            cv2.circle(obs, (x, y), 1 , (0, 0, 255), -1)
        

        # # Show a higher resolution window
        obs = cv2.resize(obs, (width*2, height*2), interpolation=cv2.INTER_NEAREST)
        
        # Run all frames in a loop without expecting a key press
        cv2.imshow('Frame', obs)
        # cv2.waitKey(1)  # Display each frame for 1 ms
        

        window_name = 'Frame'
        cv2.imshow('Frame', obs)
        key = cv2.waitKey(0)  # Wait for a key press to move
        if key == 27:
            break
    
    cv2.destroyAllWindows()


        
    # Save new dataframe to a new text file in the same directory as the original text file with _with_relationships appended to the filename
    new_text_file_path = text_file_path.replace(".txt", "_with_relationships.txt")
    # Remove gaze_positions column from the dataframe before saving
    gaze_df = gaze_df.drop(columns=['gaze_positions'])
    gaze_df.to_csv(new_text_file_path, index=False)

    print(f"Saved updated gaze data with relationships to {new_text_file_path}")
    
