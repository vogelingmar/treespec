import pytest
import os
from treespec.utils.image_tools import select_rgb_images, extract_pano_faces, find_all_trees, create_dataset
from treespec.utils.matching_tools import create_dictionary

testpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_select_rgb_images():
    input_dir = os.path.join(testpath, "mock/essen/rgb")
    output_dir = os.path.join(testpath, "mock/temp/pictures/selected_rgb")
    image_type = "jpg"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.removedirs(output_dir)

    select_rgb_images(input_dir, output_dir, image_type)

    output = os.listdir(output_dir)
    assert len(output) == 32
    for file in output:
        assert file.endswith("_rgb_left.jpg") or file.endswith("_rgb_right.jpg")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(output_dir, file))    
    os.removedirs(output_dir)


def test_extract_pano_faces():
    input_dir = os.path.join(testpath, "mock/essen/run_70/panos_70")
    id_output_dir = os.path.join(testpath, "mock/temp/pictures/id_70")
    sem_output_dir = os.path.join(testpath, "mock/temp/pictures/sem_70")
    color_output_dir = os.path.join(testpath, "mock/temp/pictures/rgb_70")
    sem_id_input_file_type = "tif"
    rgb_input_file_type = "jpg"
    output_file_type = "png"
    run_number = "70"
    apply_center_crop = False
    id_filter = "segmentid"
    sem_filter = "semanticclass"

    # test for segmentid images
    if not os.path.exists(id_output_dir):
        os.makedirs(id_output_dir)
    else:
        for file in os.listdir(id_output_dir):
            os.remove(os.path.join(id_output_dir, file))
        os.removedirs(id_output_dir)

    extract_pano_faces(input_dir, id_output_dir, sem_id_input_file_type, output_file_type, run_number, apply_center_crop, id_filter)

    output = os.listdir(id_output_dir)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"{id_filter}_left.{output_file_type}") or file.endswith(f"{id_filter}_right.{output_file_type}")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(id_output_dir, file))
    os.removedirs(id_output_dir)

    #test for semanticclass images
    if not os.path.exists(sem_output_dir):
        os.makedirs(sem_output_dir)
    else:
        for file in os.listdir(sem_output_dir):
            os.remove(os.path.join(sem_output_dir, file))
        os.removedirs(sem_output_dir)

    extract_pano_faces(input_dir, sem_output_dir, sem_id_input_file_type, output_file_type, run_number, apply_center_crop, sem_filter)

    output = os.listdir(sem_output_dir)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"{sem_filter}_left.{output_file_type}") or file.endswith(f"{sem_filter}_right.{output_file_type}")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(sem_output_dir, file))
    os.removedirs(sem_output_dir)

    # test for rgb images
    if not os.path.exists(color_output_dir):
        os.makedirs(color_output_dir)
    else:
        for file in os.listdir(color_output_dir):
            os.remove(os.path.join(color_output_dir, file))
        os.removedirs(color_output_dir)

    extract_pano_faces(input_dir, color_output_dir, rgb_input_file_type, output_file_type, run_number, apply_center_crop)

    output = os.listdir(color_output_dir)
    assert len(output) > 0
    for file in output:
        assert file.endswith(f"rgb_left.{output_file_type}") or file.endswith(f"rgb_right.{output_file_type}")
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(color_output_dir, file))
    os.removedirs(color_output_dir)


def test_find_all_trees():
    color_dir = os.path.join(testpath, "mock/essen/run_70/rgb_70")
    segmentid_dir = os.path.join(testpath, "mock/essen/run_70/id_70")
    semantic_dir = os.path.join(testpath, "mock/essen/run_70/sem_70")
    trees_output_dir = os.path.join(testpath, "mock/temp/pictures/trees_70")
    barks_output_dir = os.path.join(testpath, "mock/temp/pictures/barks_70")
    tree_attributes_dict = create_dictionary(os.path.join(testpath, "mock/essen/run_70/inventory_70/matched_output"))
    input_file_type = "png"

    # test for trees
    if not os.path.exists(trees_output_dir):
        os.makedirs(trees_output_dir)
    else:
        for file in os.listdir(trees_output_dir):
            os.remove(os.path.join(trees_output_dir, file))
        os.removedirs(trees_output_dir)

    find_all_trees(segmentid_dir=segmentid_dir,color_dir= color_dir,output_dir= trees_output_dir, tree_attributes_dict= tree_attributes_dict, cover= "tree", semantic_dir =semantic_dir, input_file_type=input_file_type)


    output = os.listdir(trees_output_dir)
    assert len(output) > 0
    for file in output:
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(trees_output_dir, file))
    os.removedirs(trees_output_dir)


    # test for barks
    if not os.path.exists(barks_output_dir):
        os.makedirs(barks_output_dir)
    else:
        for file in os.listdir(barks_output_dir):
            os.remove(os.path.join(barks_output_dir, file))
        os.removedirs(barks_output_dir)

    find_all_trees(segmentid_dir=segmentid_dir,color_dir= color_dir,output_dir= barks_output_dir, tree_attributes_dict= tree_attributes_dict, cover= "bark", semantic_dir =semantic_dir, input_file_type=input_file_type)


    output = os.listdir(barks_output_dir)
    assert len(output) > 0
    for file in output:
        parts = file.split("_")
        assert len(parts) == 3
        assert parts[0].isdigit()
        os.remove(os.path.join(barks_output_dir, file))
    os.removedirs(barks_output_dir)

def test_create_dataset(): 
    output_dataset_dir = os.path.join(testpath, "mock/temp/pictures/trees_70")
    input_trees_dir = os.path.join(testpath, "mock/essen/run_70/trees_70")
    
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)
    else:
        for dir in os.listdir(output_dataset_dir):
            dir_path = os.path.join(output_dataset_dir, dir)
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
            os.removedirs(dir_path)
        os.removedirs(output_dataset_dir)

    create_dataset(input_trees_dir=input_trees_dir, output_dataset_dir=output_dataset_dir, only_copy=True)
    
    assert os.path.exists(output_dataset_dir)
    dataset = os.listdir(output_dataset_dir)
    assert len(dataset) == 3
    for dir in dataset:
        dir_path = os.path.join(output_dataset_dir, dir)
        for file in os.listdir(dir_path):
            assert file.endswith(".png")
            parts = file.split("_")
            assert len(parts) == 3
            assert parts[0].isdigit()
            os.remove(os.path.join(dir_path, file))
        os.removedirs(dir_path)