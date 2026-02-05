import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from srlearn import Database, Background
import string
import os

# --- Configurations ---
THRESHOLD = 0.70
# Define primitive actions
primitive_actions = ["noop","fire","up","right","left","down"]

# modes (reused mostly for background setup, though not strictly needed for facts generation)
modes=["aboveOfDiver(+state, +diver).", "aboveOfEnemy(+state, +enemy).", "aboveOfMissile(+state, +missile). ", "aboveOfSubmarine(+state, +submarine).",
       "aboveWater_surface(+state).", "belowOfDiver(+state, +diver).", "belowOfEnemy(+state, +enemy).", "belowOfMissile(+state, +missile).",
       "belowOfSubmarine(+state, +submarine).", "belowWater_surface(+state).", "diversEmpty(+state).", "diversNotfull(+state).", "diversfull(+state).",
       "enemyFacingLeft(+state, +submarine).", "enemyFacingRight(+state, +submarine).", "facingLeft(+state).", "facingRight(+state).",
       "leftOfDiver(+state, +diver).", "leftOfEnemy(+state, +enemy).", "leftOfMissile(+state, +missile).", "leftOfSubmarine(+state, +submarine).",
       "nearbyDiver(+state, +diver).", "nearbyEnemy(+state, +enemy).", "nearbyMissile(+state, +missile).", "nearbySubmarine(+state, +submarine).",
       "oxygenOk(+state).", "rightOfDiver(+state, +diver).", "rightOfEnemy(+state, +enemy).", "rightOfMissile(+state, +missile).",
       "rightOfSubmarine(+state, +submarine).", "sameLevelAsDiver(+state, +diver).", "sameLevelAsEnemy(+state, +enemy).",
       "sameLevelAsMissile(+state, +missile).", "sameLevelAsSubmarine(+state, +submarine).",
       "visibleDiver(+state, -diver).", "visibleEnemy(+state, -enemy).", "visibleEnemySubmarine(+state, -submarine).",
       "visibleMissile(+state, -missile).", "action(+state, #name)."]

bridgers = ["vissibleMissile/2", "vissibleEnemy/2", "vissibleEnemySubmarine/2", "vissibleDiver/2"]
bridgers = [bridger.lower() for bridger in bridgers]
modes = [mode.lower() for mode in modes]

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Process relationship file for Teacher Model")
parser.add_argument("--file", type=str, default="", help="Relationship file")
parser.add_argument("--all", type=bool, default=False, help="Process all actions together")

args = parser.parse_args()

# --- Base Directory Setup ---
if args.all:
    base_dir = "data/seaquest/all_teacher"  # TARGETING TEACHER DIR
else:
    # Fallback to single_t teacher dir if needed, but primary use is --all
    print(file)
    file = args.file.split("/")[-1]
    file = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2]
    base_dir = "data/seaquest/single_t_teacher/"+file
print(f"Processing file: {args.file}")
print(f"Base Directory: {base_dir}")


# --- Helper Function for InRadius Replacement ---
def cleanup_inradius_name(objtype):
    # User Request: inradiusenemy1 -> inradiusenemy, inradiusenemymissile -> inradiusmissile
    # The extraction logic typically yields objtype from e.g. "enemy1" is "enemy".
    # "enemymissile1" -> "enemymissile".
    
    if objtype == "enemymissile":
        return "missile"
    if objtype == "enemy":
        return "enemy"
    # Fallback/Pass-through
    return objtype

def process_and_write_facts(df_subset, output_file, generate_inradius=True):
    """
    Writes facts to output_file.
    If generate_inradius is True, it ADDS inradius predicates based on logical derivation
    and applies requested replacements.
    """
    with open(output_file, 'w') as f:
        prev_state = ""
        prev_objnum = ""
        
        for _, row in df_subset.iterrows():
            rels = str(row["relationships"])
            weights = str(row["predicate_weights"])
            
            # For Test set, we might not have weights? 
            # Check if weights exist. If not, assume weight 1.0 needed? 
            # Or usually test set just needs existence.
            # But the original code relied on weights > THRESHOLD for inradius generation.
            # If test set has weights, use them. If not, what to do?
            # Looking at original code: test writes facts without weight check.
            # But user wants inradius in test facts.
            # If test csv has weights, we use them.
            
            weights_list = []
            if weights != "nan":
                weights_list = weights.split(" ")
            
            if rels != "nan":
                s_id = "s" + str(row["frameid"].replace("_",""))
                rels_list = rels.split(" , ")
                rels_list[-1] = rels_list[-1].replace(" ,","")
            
                
                # If weights missing (possible in some test scenarios?), pad with 1.0 or skip?
                # Original code for test facts just iterates rels without weights.
                # Here we need weights for thresholding inradius?
                # If weights list is empty, we can't threshold.
                # Assuming test data usually has the same structure as train data in this dataset.
                
                # Parse weights into a dictionary
            
                weight_map = {}
                if weights != "nan":
                    # Create a mapping weight_map from predicate to weight
                    # Example: "1.000 1.000 1.000 0.913 0.913 1.000" -> {"facingRight": "1.000", "visibleEnemy(enemy_1)": "0.913", ...}
                    for i, rel in enumerate(rels_list):
                        weight_map[rel] = weights_list[i]
                   
                
                # Debug print for first few rows
                if len(weight_map) > 0 and len(df_subset) < 2000 and s_id == "sRZ2461867161": 
                    print(f"DEBUG: Processing {s_id}")
                    print(f"DEBUG: Weight Map: {weight_map}")

                # Fact Generation Logic
                # 1. Write ALL original predicates first (cleaning trailing commas).
                # 2. Iterate through predicates to identify objects and their weights.
                # 3. For non-visible predicates, check weight W.
                #    If W > 0.7: Write inradiusType(S, O) AND visibleType(S, O).
                #    Else: Write visibleType(S, O).
                
                generated_facts = set()

                for i, rel in enumerate(rels_list):
                    rel = rel.strip()
                    while rel.endswith(' ,'):
                        rel = rel[:-2].strip()
                    rel = rel.strip(",")
                    if not rel: continue
                    
                    # See if visible in rel, if so  find the object of that rel and check any predicate in rel_list that uses that object. store it in alternate_rel and use it to get weight of that predicate.
                    alternate_rel = None
                    if "visible" in rel:
                        object_name = rel.split("(")[1].split(")")[0]
                        for rel_ in rels_list:
                            if object_name in rel_ and "visible" not in rel_ and "Facing" not in rel_:
                                alternate_rel = rel_
                                if alternate_rel.endswith(" ,"):
                                    alternate_rel = alternate_rel[:-2]
                                    alternate_rel = alternate_rel.strip(",")
                                break
                        
                   
                        
                    original_rel = rel
                    if alternate_rel:
              
                        w = weight_map.get(alternate_rel)
                    else:
                        w = weight_map.get(original_rel)
                   
                  
                    # Format standard fact
                    if "(" not in rel:
                        rel = rel+"()"
                    rel = rel.replace("(","(" + s_id + ",")
                    rel = rel.replace(",)", ")")
                    if not rel.endswith("."):
                        rel += "."
                    rel = rel.lower()
                    rel = rel.replace("_", "")
                    
                    f.write(rel + "\n")
                    
                    # --- InRadius / Visible Generation ---
                    if generate_inradius:
                        # "checking the object weight for any predicate other than visible predicate"
                        if "visible" in rel: 
                            continue
                            
                        # Extract object from relation
                        # Expecting binary relations like leftOfEnemy(sID, enemy0)
                        # We need to parse 'rel' which is now like "leftofenemy(srz...,enemy0)."
                        
                        if rel.count(",") == 1:
                            try:
                                parts = rel.split("(")
                                if len(parts) > 1:
                                    args_part = parts[1].split(",")
                                    if len(args_part) >= 2:
                                        state = args_part[0].strip()
                                        obj_num = args_part[1].split(")")[0].strip()
                                        
                                        # Sanity checks
                                        if not obj_num or state == obj_num: continue
                                        if obj_num.startswith('s') and (obj_num in state or state in obj_num): continue
                                        
                                        # Derive Type
                                        objtype = obj_num
                                        while objtype and objtype[-1].isdigit():
                                            objtype = objtype[:-1]
                                        if objtype.endswith('_'): objtype = objtype[:-1]
                                        
                                        clean_type = cleanup_inradius_name(objtype)
                                        
                                        # Allowed types for this logic
                                        allowed = ["enemy", "enemysubmarine", "diver", "missile"]
                                        if clean_type not in allowed: continue
                                        
                                        # Threshold Logic
                                        # Check weight of the SPATIAL predicate (original_rel)
                                        # If > 0.7: inradius + visible
                                        # Else: visible
                                        
                                        weight_val = float(w)
                                        
                                        # Construct predicates
                                        pred_inradius = f"inradius{clean_type}({state},{obj_num}).\n"
                                        pred_visible = f"visible{clean_type}({state},{obj_num}).\n"
                                        
                                        if weight_val > 0.7:
                                            if pred_inradius not in generated_facts:
                                                f.write(pred_inradius)
                                                generated_facts.add(pred_inradius)
                                            if pred_visible not in generated_facts:
                                                f.write(pred_visible)
                                                generated_facts.add(pred_visible)
                                        else:
                                            # "If not just just the visibleObject predicate should be added"
                                            if pred_visible not in generated_facts:
                                                f.write(pred_visible)
                                                generated_facts.add(pred_visible)
                                                
                            except Exception as e:
                                pass
                    
                    # Standard fact formatting
                    if "(" not in rel:
                        rel = rel+"()"
                    rel = rel.replace(",)", ")")
                    if not rel.endswith("."):
                        rel += "."
                    rel = rel.lower()
                    rel = rel.replace("_", "")
                    
                    # Write standard fact
                    f.write(rel + "\n")
                    
                    # --- InRadius Logic ---
                    if generate_inradius:
                         # If visible in rel ignore for inradius generation
                        if "visible" in rel:
                            continue

                        # Logic from train generation:
                        if "visible" not in rel and rel.count(",") == 1:
                            try:
                                parts = rel.split("(")
                                if len(parts) > 1:
                                    args_part = parts[1].split(",")
                                    if len(args_part) >= 2:
                                        state = args_part[0]
                                        obj_num = args_part[1].split(")")[0]
                                        
                                        # Deduplicate per object/state sequence
                                        if prev_state == state and prev_objnum == obj_num:
                                            pass
                                        else:
                                            prev_state = state
                                            prev_objnum = obj_num
                                            
                                            # Derive objtype
                                            # obj_num e.g. "enemy1", "missile2"
                                            # remove last char if digit? 
                                            # The original code did: objtype = obj_num.split(")")[0][:-1]
                                            # But wait, obj_num is usually "enemy1".
                                            if obj_num[-1].isdigit():
                                                objtype = obj_num[:-1]
                                            else:
                                                objtype = obj_num
                                            
                                            # Apply Replacements
                                            clean_type = cleanup_inradius_name(objtype)
                                            
                                            # Threshold check
                                            if float(w) > THRESHOLD:
                                                f.write(f"inradius{clean_type}({state},{obj_num}).\n")
                            except Exception as e:
                                # Fallback or logging if parse fails
                                pass

# --- New Parsing Logic for relationships.txt ---
# Format analysis:
# Columns seem tab separated? Header inspection needed or assumption based on previous file content.
# The user said "Use relationships.txt".
# Sample line: 
# RZ_9656617_17 <tab> 0 <tab> ... <tab> [(...)] <tab> rels_string <tab> weights_string <tab> filename
# It seems there is no standard CSV header in the sample output, or it's complex.
# However, the user provided file `relationships.txt` in `data/seaquest/gaze_data_tmp/` seems to have columns.
# Let's write a robust parser for this specific file structure.

def parse_relationships_file(filepath):
    """
    Parses relationships.txt which has a custom tab-separated format.
    Returns a DataFrame-like structure (list of dicts) with:
    - frameid
    - relationships (string)
    - predicate_weights (string)
    - action (int) - derived if present, else 0? 
      Wait, `relationships.txt` sample shows "0" in 2nd column? 
      "RZ_9656617_17 0 ..." -> 0 might be score or action?
      The sample output shows:
      RZ... <tab> 0 <tab> 54330 <tab> ...
      Let's assume column 2 is action? No, typically score.
      The previous file `54_RZ...` had `frameid,episode_id,score...,action`.
      
      Let's look at the file head output again.
      RZ_9656617_17   0       54330   50.0    0.0     4       [...]   player_0 , ...  facingLeft , ...        1.000 1.000 ...
      
      Col 0: Frame ID
      Col 1: ? (0)
      Col 2: ? (54330)
      Col 3: ? (50.0)
      Col 4: ? (0.0)
      Col 5: Action (4) -> This seems valid (actions are 0-5 typically, 4 is int).
      Col 6: Objects List [...]
      Col 7: Objects String "player_0 , ..."
      Col 8: Relationships "facingLeft , ..."
      Col 9: Goals "retrieve_diver"
      Col 10: Weights "1.000 1.000 ..."
      
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split('\t')
            if len(parts) < 10: 
                # Maybe header or malformed
                continue
            
            # Heuristic to skip header if "frameid" is in line
            if "frameid" in line.lower():
                continue

            try:
                frame_id = parts[0].strip()
                action = int(parts[5].strip())
                relationships = parts[8].strip()
                # Col 10 was empty, weights at 11
                if len(parts) > 11:
                    weights = parts[11].strip()
                else:
                    weights = ""
                
                data.append({
                    "frameid": frame_id,
                    "action": action,
                    "relationships": relationships,
                    "predicate_weights": weights
                })
            except Exception as e:
                # print(f"Skipping line due to error: {e}")
                continue
    
    return pd.DataFrame(data)


# --- Main Processing ---

# Ensure directories exist
for action in primitive_actions:
    os.makedirs(f"{base_dir}/{action}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/{action}/test", exist_ok=True)

# Read Data using custom parser
print(f"Parsing {args.file} as relationships.txt format...")
df = parse_relationships_file(args.file)

# Filter actions <= 5 (primitive)
df = df[df['action'] <= 5]

# Split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# --- Process Train Facts ---
print("Regenerating Train Facts (Teacher)...")
for pa in primitive_actions:
    out_file = f"{base_dir}/{pa}/train/train_facts.txt"
    process_and_write_facts(train, out_file, generate_inradius=True)
    
    # Dedup logic
    with open(out_file, 'r') as f_read:
        lines = f_read.readlines()
    seen = set()
    deduped_lines = []
    for line in lines:
        if line in seen: continue
        seen.add(line)
        deduped_lines.append(line)
    with open(out_file, 'w') as f_write:
        f_write.writelines(deduped_lines)

# --- Process Test Facts ---
print("Regenerating Test Facts (Teacher)...")
for pa in primitive_actions:
    out_file = f"{base_dir}/{pa}/test/test_facts.txt"
    process_and_write_facts(test, out_file, generate_inradius=True)
    
    # Dedup
    with open(out_file, 'r') as f_read:
        lines = f_read.readlines()
    seen = set()
    deduped_lines = []
    for line in lines:
        if line in seen: continue
        seen.add(line)
        deduped_lines.append(line)
    with open(out_file, 'w') as f_write:
        f_write.writelines(deduped_lines)

print("Preprocessing complete.")
