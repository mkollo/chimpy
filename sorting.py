#    ___ _  _ ___ __  __ _____   __
#   / __| || |_ _|  \/  | _ \ \ / /
#  | (__| __ || || |\/| |  _/\ V / 
#   \___|_||_|___|_|  |_|_|   |_| 
# 
# Copyright (c) 2020 Mihaly Kollo. All rights reserved.
# 
# This work is licensed under the terms of the MIT license.  
# For a copy, see <https://opensource.org/licenses/MIT>.                                 

def write_probe_file(filename, coords, radius):
    with open(filename, "w") as fid:
        fid.write("total_nb_channels = "+str(coords.shape[0])+"\n")
        fid.write("radius = "+str(radius)+"\n")        
        fid.write("channel_groups = {\n")
        fid.write("\t1: {\n")
        fid.write("\t\t'channels': list(range("+srt(coords.shape[0])+")),\n")
        fid.write("\t\t'graph': [],\n")
        fid.write("\t\t'geometry': {\n")
        for i in range(coords.shape[0]):
            fid.write("\t\t\t"+str(i)+":  [  "+', '.join([str(c) for c in coords[i,:]])+"],\n")
        fid.write("\t\t}\n")
        fid.write("\t}\n")
        fid.write("}\n")


        