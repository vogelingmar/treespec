import os
from pathlib import Path

from pointtorch import read
from pointtorch.operations.numpy import make_labels_consecutive

input_dir = Path("/home/ingmar/Downloads/test")
output_dir = Path("/home/ingmar/Downloads/tree_filtering_intensity")
output_dir.mkdir(exist_ok=True, parents=True)

for idx, file in enumerate(sorted(os.listdir(input_dir))):
    print(file)
    file_path = input_dir / file

point_cloud = read(file_path)
print("point_cloud", point_cloud.columns)
remapped_instance_ids = make_labels_consecutive(point_cloud["segmentidpredicted"].to_numpy(), ignore_id=-1, start_id=1)
remapped_instance_ids[remapped_instance_ids == -1] = 0

assert remapped_instance_ids.max() < 2 ** 16, "Too many instances to store them with 16 bit."

point_cloud.drop(["intensity", "red", "green", "blue", "semclassidpredicted", "specificclassidpredicted", "segmentidpredicted", "distancetodtm"], axis=1, inplace=True)
point_cloud["intensity"] = remapped_instance_ids
# point_cloud.rename({"segmentidpredicted": "intensity"}, axis=1, inplace=True)

print("output_dir / file", output_dir / file)

point_cloud.to(output_dir / file)