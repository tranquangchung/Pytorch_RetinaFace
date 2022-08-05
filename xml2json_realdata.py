import json
import pdb
import glob

f_rikkei = open("rikkei.txt", "w")

jsons = glob.glob("./xml_json/*.json")
for json_file in jsons:
    # Opening JSON file
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
        name_folder = data['annotations']['meta']['task']['name']
        data = data['annotations']['image']
        for i in range(len(data)):
            try:
                stop = False
                writelist = []
                name = data[i]["_name"]
                width = data[i]["_width"]
                height = data[i]["_height"]
                box = data[i]["box"]
                if not isinstance(box, list):
                    box = [box]
                len_box = len(box)
                first_line = "# {0}/{1}\n".format(name_folder, name)
                writelist.append(first_line)
                if name == "timed_20220726183924_40_4327.jpg":
                    stop = True
                if "points" in data[i]:
                    points = data[i]["points"]
                    for i_b in range(len_box):
                        l_eye = None
                        r_eye = None
                        nose = None
                        l_mouth = None
                        r_mouth = None
                        box_x1, box_y1, box_x2, box_y2 = int(float(box[i_b]["_xtl"])), int(float(box[i_b]["_ytl"])), int(float(box[i_b]["_xbr"])), int(float(box[i_b]["_ybr"]))
                        if box[i_b]['attribute'][0]['_name'] == 'ID':
                            id_box = box[i_b]['attribute'][0]['__text']
                        if box[i_b]['attribute'][1]['_name'] == 'Hard':
                            hard = box[i_b]['attribute'][1]['__text']
                            hard = 1 if hard == 'true' else 0
                        for j in range(len(points)):
                            point = points[j]
                            id_point = point['attribute'][-1]['__text']
                            if point["_label"] == "Left eye" and id_point == id_box:
                                l_eye = point['_points'].split(",")
                            if point["_label"] == "Right eye" and id_point == id_box:
                                r_eye = point['_points'].split(",")
                            if point["_label"] == "Nose" and id_point == id_box:
                                nose = point['_points'].split(",")
                            if point["_label"] == "Left mouth" and id_point == id_box:
                                l_mouth = point['_points'].split(",")
                            if point["_label"] == "Right mouth" and id_point == id_box:
                                r_mouth = point['_points'].split(",")
                        if l_eye and r_eye and nose and l_mouth and r_mouth:
                            line = "{0} {1} {2} {3} {4} {5} 0.0 {6} {7} 0.0 {8} {9} 0.0 {10} {11} 0.0 {12} {13} 0.0 1.0 {14}\n".format(
                            box_x1, box_y1, box_x2, box_y2,
                            l_eye[0], l_eye[1], # left eye
                            r_eye[0], r_eye[1], # right eye
                            nose[0], nose[1], # nose
                            l_mouth[0], l_mouth[1], # left mouth
                            r_mouth[0], r_mouth[1], # right mouth
                            hard,
                            )
                        else:
                            line = "{0} {1} {2} {3} {4} {5} 0.0 {6} {7} 0.0 {8} {9} 0.0 {10} {11} 0.0 {12} {13} 0.0 1.0 {14}\n".format(
                                box_x1, box_y1, box_x2, box_y2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, hard)
                        writelist.append(line)
                else:
                    for i_b in range(len_box):
                        box_x1, box_y1, box_x2, box_y2 = int(float(box[i_b]["_xtl"])), int(float(box[i_b]["_ytl"])), int(float(box[i_b]["_xbr"])), int(float(box[i_b]["_ybr"]))
                        if box[i_b]['attribute'][0]['_name'] == 'ID':
                            id_box = box[i_b]['attribute'][0]['__text']
                        if box[i_b]['attribute'][1]['_name'] == 'Hard':
                            hard = box[i_b]['attribute'][1]['__text']
                            hard = 1 if hard == 'true' else 0
                        line = "{0} {1} {2} {3} {4} {5} 0.0 {6} {7} 0.0 {8} {9} 0.0 {10} {11} 0.0 {12} {13} 0.0 1.0 {14}\n".format(
                        box_x1, box_y1, box_x2, box_y2,
                        0, 0, # left eye
                        0, 0, # right eye
                        0, 0, # nose
                        0, 0, # left mouth
                        0, 0, # right mouth
                        hard,
                        )
                        writelist.append(line)

                for l in writelist:
                    f_rikkei.write(l)
            except Exception as e:
                if e.args[0] != "box":
                    print(i, name)
                    print(e)
