import json
import pdb
 
# Opening JSON file
with open('./convertjson.json', 'r') as json_file, open('./mafa2.txt', 'w') as mafa:
    data = json.load(json_file)
    data = data['annotations']['image']
    for i in range(len(data)):
        try:
            stop = False
            writelist = []
            name = data[i]["_name"]
            width = data[i]["_width"]
            height = data[i]["_height"]
            box = data[i]["box"]
            len_box = len(box)
            points = data[i]["points"]
            first_line = "# mafa/{0}\n".format(name)
            writelist.append(first_line)
            # if name == "train_00002022.jpg":
            #     pdb.set_trace()
            #     stop = True
            for i in range(len_box):
                l_eye = None
                r_eye = None
                nose = None
                l_mouth = None
                r_mouth = None
                box_x1, box_y1, box_x2, box_y2 = int(float(box[i]["_xtl"])), int(float(box[i]["_ytl"])), int(float(box[i]["_xbr"])), int(float(box[i]["_ybr"]))
                id_box = box[i]['attribute']['__text']
                for j in range(len(points)):
                    point = points[j]
                    id_point = point['attribute']['__text']
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
                line = "{0} {1} {2} {3} {4} {5} 0.0 {6} {7} 0.0 {8} {9} 0.0 {10} {11} 0.0 {12} {13} 0.0 1.0\n".format(
                box_x1, box_y1, box_x2, box_y2,
                l_eye[0], l_eye[1], # left eye
                r_eye[0], r_eye[1], # right eye
                nose[0], nose[1], # nose
                l_mouth[0], l_mouth[1], # left mouth
                r_mouth[0], r_mouth[1], # right mouth
                )
                # if stop:
                #     pdb.set_trace()
                #     # print(line)
                writelist.append(line)
            for l in writelist:
                mafa.write(l)
        except Exception as e:
            if e.args[0] != "box":
                print(i, name)
                print(e)
