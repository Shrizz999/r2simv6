#!/usr/bin/env python3
import sys
import os
import xml.etree.ElementTree as ET

def convert_sdf_to_urdf(sdf_path, urdf_path, pkg_name):
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    # Handle standard SDF structure
    model = root.find('model') if root.find('model') is not None else root
    model_name = model.get('name', 'DDRena')

    urdf_root = ET.Element("robot", name=model_name)
    
    # Dummy world link
    ET.SubElement(urdf_root, "link", name="world")
    
    for link in model.findall('link'):
        link_name = link.get('name')
        
        # Parse Pose
        pose = link.find('pose')
        pose_text = pose.text if pose is not None else "0 0 0 0 0 0"
        vals = [float(x) for x in pose_text.split()]
        xyz = f"{vals[0]} {vals[1]} {vals[2]}"
        rpy = f"{vals[3]} {vals[4]} {vals[5]}"
        
        # Link
        urdf_link = ET.SubElement(urdf_root, "link", name=link_name)
        
        # Visual
        visual_sdf = link.find('visual')
        if visual_sdf is not None:
            visual_urdf = ET.SubElement(urdf_link, "visual")
            geometry_urdf = ET.SubElement(visual_urdf, "geometry")
            mesh_sdf = visual_sdf.find('geometry').find('mesh')
            
            if mesh_sdf is not None:
                mesh_urdf = ET.SubElement(geometry_urdf, "mesh")
                uri = mesh_sdf.find('uri').text
                
                # UPDATED: Replaces 'model://DDRena' or 'model://ARENA' with package path
                new_uri = uri.replace(f"model://{model_name}", f"package://{pkg_name}/models/{model_name}")
                # Fallback if the SDF still says ARENA
                new_uri = new_uri.replace("model://ARENA", f"package://{pkg_name}/models/{model_name}")
                
                mesh_urdf.set("filename", new_uri)
                
                scale = mesh_sdf.find('scale')
                if scale is not None:
                    mesh_urdf.set("scale", scale.text)

            # Material
            mat = ET.SubElement(visual_urdf, "material", name="grey")
            ET.SubElement(mat, "color", rgba="0.7 0.7 0.7 1.0")

        # Joint
        joint = ET.SubElement(urdf_root, "joint", name=f"{link_name}_joint", type="fixed")
        ET.SubElement(joint, "parent", link="world")
        ET.SubElement(joint, "child", link=link_name)
        ET.SubElement(joint, "origin", xyz=xyz, rpy=rpy)

    tree = ET.ElementTree(urdf_root)
    ET.indent(tree, space="  ", level=0)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    # Args: input_sdf, output_urdf, package_name
    convert_sdf_to_urdf(sys.argv[1], sys.argv[2], sys.argv[3])
