import xml.etree.cElementTree as ET
import xml


def create_topology_xml(name, visible_layers, *hidden_layers):
    topology = ET.Element("topology")
    vls = ET.SubElement(topology, "visible_layers")
    ET.SubElement(vls, "input_layer").text = "%d" % visible_layers[0]
    ET.SubElement(vls, "output_layer").text = "%d" % visible_layers[1]

    hls = ET.SubElement(topology, "hidden_layers")

    for hl in hidden_layers:
        ET.SubElement(hls, "hidden_layer").text = "%d" % hl

    tree = ET.ElementTree(topology)
    tree.write("userspace/saved_nns/{}/{}.xml".format(name, name))


def parse_topology_xml(filepath):
    root = xml.etree.ElementTree.parse(filepath).getroot()
    visible = root[0]
    hidden = root[1]
    visible_layers = (int(visible.find('input_layer').text), int(visible.find('output_layer').text))
    hidden_layers = []

    for h in hidden.iter('hidden_layer'):
        hidden_layers.append(int(h.text))

    return visible_layers, hidden_layers
