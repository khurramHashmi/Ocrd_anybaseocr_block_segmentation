import numpy
import xml.etree.ElementTree as ET
import glob
import argparse
import os
import pprint
import json
from io import StringIO
from skimage.measure import approximate_polygon
parser = argparse.ArgumentParser(description="generate a JSON file from the ground-truth provided in XML")

parser.add_argument("-n", "--namespace", default="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15", help="namespace for the XML file")
parser.add_argument("-o", "--output", default="temp", help="path to store output files")
parser.add_argument("-input", help="path to input xml data")

args = parser.parse_args()
ns = "{"+args.namespace+"}"

if args.output=="temp":
   output = args.input


# this function parse XML and creates a dictionary having information for all polygons present in an image.
def write_image_info_json(data,gt): 

    gt = gt.getroot()
    ns = gt.tag.split('PcGts')[0]
    
    for region in gt.iter(ns+args.mode):
        all_x_values = list()
        all_y_values = list()
        #get region coordinates
        
        if "type" in region.attrib: 
            type_tag = region.attrib["type"]
            #fine region
            if type_tag != "other":
                coords = region.find(ns+"Coords")
                last_x, last_y = 0,0   
                if "points" in coords.attrib:
                    points = coords.attrib["points"].split(" ")

                #Storing coordinates,image_name and class of block in JSON
                    for p in points:
                        all_y_values.append(int(p.split(",")[1]))
                        all_x_values.append(int(p.split(",")[0]))
                else:
                    for coord_points in coords: 
                        sum_val = (last_x - int(coord_points.attrib["x"])) + (last_y - int(coord_points.attrib["y"]))   
                        if(type_tag == "paragraph"):
                            #if(sum_val>200 or sum_val<-200):                                
                            last_x = int(coord_points.attrib["x"])
                            last_y = int(coord_points.attrib["y"])
                            #print(sum_val, last_x, last_y)
                            all_x_values.append(int(coord_points.attrib["x"]))
                            all_y_values.append(int(coord_points.attrib["y"]))
                        else:    
#                             if(sum_val>10 or sum_val<-10):
                            last_x = int(coord_points.attrib["x"])
                            last_y = int(coord_points.attrib["y"])
                            #print(sum_val, last_x, last_y)
                            all_x_values.append(int(coord_points.attrib["x"]))
                            all_y_values.append(int(coord_points.attrib["y"]))
                point = numpy.zeros((len(all_x_values),2))
                point[:,0] = all_x_values
                point[:,1] = all_y_values
                
                #Restricting the polygon coordinates to 20 at max to avoid memory load
                tolerance = 0.0
                if len(point) > 20:
                    new_point = point
                    while len(new_point) > 20 and tolerance < 100 :
                        tolerance += 1.0
                        new_point = approximate_polygon(point, tolerance=tolerance)
                    point = new_point

                data.append({  
                    'block_class': type_tag,
                    'all_x_values': list(point[:,0]),
                    'all_y_values': list(point[:,1])
                })

    return data

print("Generating json data ...")
basenames = list()
count=0

#Iterating over all the XML files present in the input directory
for file_path in glob.glob(args.input+"/*.xml"):
    
    #generate_gt_path
    basename = os.path.basename(file_path).split(".")[0]
    basenames.append(basename)
    data = list()
    gt_xml_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path))
    
    #load xml
    gt = ET.parse(gt_xml_path)
    data = write_image_info_json(data,gt)
    count+=1

    #writing the content of each XML file into a JSON file
    #Each XML file or Image should have it's own corresponding JSON file
    with open(os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]+".json"), 'w') as outfile:  
        json.dump(data, outfile, indent=4)
        
#One JSON file containing names of all the JSON files.
#This helps in loading one JSON file at a time which avoid large memory consumptions.
with open('base_names.json', 'w') as outfile:  
    json.dump(basenames, outfile, indent=4)
        
print("JSON file generated")
