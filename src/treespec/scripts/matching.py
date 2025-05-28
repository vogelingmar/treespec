import shapefile
from scipy.spatial import cKDTree

class Matcher:
    def match():

        attributes_path = "/data/essen/cadastre/tree_attributes_filtered/20220905_092821_0041/20220905_092821_0041"
        cadastre_path = "/data/essen/cadastre/Essen-Baumkataster-Ausschnitt/Essen-Baumkataster-Ausschnitt"

        attributes = shapefile.Reader(attributes_path)
        cadastre = shapefile.Reader(cadastre_path)

        # Build list of (point, record) for attributes
        attribute_shape_records = attributes.shapeRecords()
        attribute_points = []
        attribute_records = []
        for shaperec in attribute_shape_records:
            for pt in shaperec.shape.points:
                attribute_points.append(pt)
                # Convert record to dict and prefix keys with 'pred_'
                pred_record = {f"pred_{k}": v for k, v in shaperec.record.as_dict().items()}
                attribute_records.append(pred_record)

        # Build list of (point, record) for cadastre
        cadastre_shape_records = cadastre.shapeRecords()
        cadastre_points = []
        cadastre_records = []
        for shaperec in cadastre_shape_records:
            for pt in shaperec.shape.points:
                cadastre_points.append(pt)
                cadastre_records.append(shaperec.record.as_dict())

        # Build KDTree for cadastre points
        cadastre_tree = cKDTree(cadastre_points)
        attribute_tree = cKDTree(attribute_points)

        # Find nearest neighbor in cadastre for each attribute point
        cad_distances, cad_indices = cadastre_tree.query(attribute_points)
        # Find nearest neighbor in attributes for each cadastre point
        att_distances, att_indices = attribute_tree.query(cadastre_points)

        # Combine records only if mutual nearest neighbors
        attributes = []
        for i, (attr_pt, cad_idx, cad_dist) in enumerate(zip(attribute_points, cad_indices, cad_distances)):
            # Check if the nearest attribute point to this cadastre point is the original attribute point
            if cad_dist <= 3000.0 and att_indices[cad_idx] == i:
                combined = {**cadastre_records[cad_idx], **attribute_records[i]}
                attributes.append(combined)
                # Optionally: print(f"Mutual match for attribute {i} and cadastre {cad_idx}")

        return attributes
    
print(Matcher.match())


