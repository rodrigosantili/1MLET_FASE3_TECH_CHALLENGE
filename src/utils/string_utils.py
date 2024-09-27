import os
import xml.etree.ElementTree as ET

from src.config import BUNDLES_DIR


def load_strings(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return {child.attrib['name']: child.text for child in root}


strings = load_strings(os.path.join(BUNDLES_DIR, 'strings.xml'))
