import re

texture_type = "snow"

with open("/Users/ste/code/active_learning/install/elohim/share/elohim/models/ground_plane/materials/scripts/snow.material") as f:
    string = f.read()

script = f"<script>\
    <uri>model://ground_plane/materials/scripts/{texture_type}.material</uri>\
    <uri>model://ground_plane/materials/textures/{texture_type}.jpg</uri>\
    <name>{texture_type}/Diffuse</name>\
</script>"

m = re.search(r'texture_unit.*?scale\s([0-9\.]+)\s([0-9\.]+).*?}', string, flags=re.S)
print(m.group(2))

