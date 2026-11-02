import json
import math
import os
import cv2 as cv
import sys
def text_process(input_path:str,min_score:float = 0.6):
    try:
        with open(input_path,'r',encoding='utf-8') as f:
            data = json.load(f)
            report = {}
            counts = {}
            counts['person'] = 0
            counts ['head'] = 0
            counts ['helmet']= 0
            review_images = []
            reviews = [] 
            errors = []
            check_image_reviews = []
            ##association,instance 数值格式检测
            try:
                for error_check in data['instances']:
                    if 'instance_id' not in error_check or 'label' not in error_check or 'detection_score' not in error_check or 'box_xyxy' not in error_check:
                        raise ValueError('invalid_instance') 
                    if error_check['label'] not in ['person','head','helmet']:
                        raise ValueError('invalid_instance')
                    score = error_check['detection_score']
                    if not isinstance(score,(int,float)):
                        raise ValueError('invalid_instance')
                    if error_check['detection_score']<0 or error_check['detection_score'] >1:
                        raise ValueError('invalid_instance')
                    if math.isinf(error_check['detection_score']) or math.isnan(error_check['detection_score']):
                        raise ValueError('invalid_instance') 
                    if isinstance(error_check['box_xyxy'],list):
                        if len(error_check['box_xyxy']) == 4:
                            box = error_check['box_xyxy']
                            if not all(type(value) in (int,float) and math.isfinite(value) 
                            for value in box):
                                raise ValueError('invalid_instance')
                            if error_check['box_xyxy'][0]<0 or error_check['box_xyxy'][1]<0 or error_check['box_xyxy'][2]<=error_check['box_xyxy'][0] or error_check['box_xyxy'][3] <= error_check['box_xyxy'][1] or error_check['box_xyxy'][2]>data['width'] or error_check['box_xyxy'][3]>data['height'] :
                                raise ValueError('invalid_instance')
                            else:
                                continue
                        else:
                            raise ValueError('invalid_instance')
                    else:
                        raise ValueError('invalid_instance')
                id = []
                for id_check in data['instances']:
                    if id_check['instance_id'] not in id:
                        id.append(id_check['instance_id'])
                    else:
                        raise ValueError('duplicate_instance_id')
                select_instances = {
                    dic['instance_id']:dic
                    for dic in data['instances']
                }
                for association_check in data['associations']:
                    if association_check['person_id'] is None:
                        raise ValueError('invalid_association')
                    else:
                        if association_check['person_id'] not in select_instances:
                            raise ValueError('invalid_association')
                        if select_instances[association_check['person_id']]['label'] != 'person':
                            raise ValueError('invalid_association')      
                    if association_check['head_id'] is not None:
                        if association_check['head_id'] not in select_instances:
                            raise ValueError('invalid_association')
                        if select_instances[association_check['head_id']]['label'] != 'head':
                            raise ValueError('invalid_association')
                    if association_check['helmet_id'] is not None:
                        if association_check['helmet_id'] not in select_instances:
                            raise ValueError('invalid_association')
                        if select_instances[association_check['helmet_id']]['label'] != 'helmet':
                            raise ValueError('invalid_association')
            except ValueError as ve:
                reason = str(ve)
                messages = {
                    'invalid_instance':'检测目标的数据不合法',
                    'duplicate_instance_id': '检测目标的instance_id重复',
                    'invalid_association':'关联记录引用的目标不存在或类别不匹配'
                }
                if reason not in messages:
                    raise
                else:
                    errors.append({'image':data['image'],'reason':reason,'message':messages[reason]})
            else:
            ##计数处理
                for check_counts in data['instances']:
                    if check_counts['detection_score']>=min_score:
                        counts[check_counts['label']]+=1
                ##复查处理
                for check_associations in data['associations']:
                    stop_helmet = False
                    stop_head = False
                    reviews_results = {}
                    if check_associations['person_id'] == None:
                        continue
                    else:
                        for check_person in data['instances']:
                            if check_person['instance_id'] == check_associations['person_id'] and check_person['label']=='person':
                                if check_person['detection_score'] >= min_score:
                                    if check_associations['head_id']==None:
                                        reviews_results['person_id']= check_associations['person_id']
                                        reviews_results['reason'] = 'head_unmatched'
                                        reviews_results['review_status'] = 'pending'
                                        reviews_results['image'] = data['image']
                                        check_image_reviews.append(1)
                                        reviews.append(reviews_results)
                                        break
                                    else:
                                        for check_head in data['instances']:
                                            if check_head['instance_id'] == check_associations['head_id'] and check_head['label'] == 'head':
                                                if check_head['detection_score']>=min_score:
                                                    if check_associations['helmet_id']==None:
                                                        reviews_results['person_id']= check_associations['person_id']
                                                        reviews_results['reason'] = 'helmet_unmatched'
                                                        reviews_results['review_status'] = 'pending'
                                                        reviews_results['image'] = data['image']
                                                        check_image_reviews.append(1)
                                                        reviews.append(reviews_results)
                                                        stop_head = True
                                                        break
                                                    else:
                                                        for check_helmet in data['instances']:
                                                            if check_helmet['instance_id'] == check_associations['helmet_id'] and check_helmet['label'] == 'helmet':
                                                                if check_helmet['detection_score']<min_score:
                                                                    reviews_results['person_id']= check_associations['person_id']
                                                                    reviews_results['reason'] = 'helmet_unmatched'
                                                                    reviews_results['review_status'] = 'pending'
                                                                    reviews_results['image'] = data['image']
                                                                    check_image_reviews.append(1)
                                                                    reviews.append(reviews_results)
                                                                    stop_helmet = True
                                                                    break
                                                                else:
                                                                    stop_helmet = True
                                                                    break
                                                        if stop_helmet == True:
                                                            stop_head = True
                                                            break                             
                                                else:
                                                    reviews_results['person_id'] = check_associations['person_id']
                                                    reviews_results['reason'] = 'head_unmatched'
                                                    reviews_results['review_status'] = 'pending'
                                                    reviews_results['image'] = data['image']
                                                    check_image_reviews.append(1)
                                                    reviews.append(reviews_results)
                                                    stop_head = True
                                                    break
                                        if stop_head == True:
                                            break                                 
                                else:
                                    break 
                if check_image_reviews:
                    review_images.append(data['image'])
            report['counts'] = counts
            report['review_images'] = review_images
            report['reviews'] = reviews
            report['errors'] = errors
    except json.JSONDecodeError:
        raise
    return report
def sign_picture(input_path:str,report:dict,min_score=0.6):
    os.makedirs('output/images',exist_ok=True)
    with open(input_path,'r',encoding='utf-8') as f:
        data = json.load(f)
        input_image_path = os.path.join('input','images',data['image'])
        try:
            check=os.path.isfile(input_image_path)
            if check is False:
                raise FileNotFoundError('image_not_found')
        except FileNotFoundError:
            report['errors'].append({
                                    'image': data['image'],
                                    'reason': 'image_not_found',
                                    'message': '图片文件不存在'
                                    })
            return False
        else:
            try:
                img = cv.imread(input_image_path)
                if img is None:
                    raise ValueError('image_read_failed')
            except ValueError:
                report['errors'].append({'image':data['image'],
                                        'reason':'image_read_failed',
                                        'message':'图片文件存在，但无法读取'})
                return False
            else:
                ## 图片标记
                person_text_positions = {}
                for check in data['instances']:
                    if check['detection_score'] >= min_score and check['label'] == 'person':
                        image_h,image_w = img.shape[:2]
                        margin = 4
                        font_scale = 2
                        x1,y1,x2,y2 = map(int,check['box_xyxy'])
                        left_up_position = tuple((x1,y1))
                        right_up_position = tuple((x2,y2))
                        rectangle = cv.rectangle(img,left_up_position,right_up_position,(255,0,0),5)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        text_sign = check['label']+' '+'#'+str(check['instance_id'])+' '+str(check['detection_score'])
                        for _ in range(60):
                            (text_w,text_h),baseline = cv.getTextSize(text_sign,font,font_scale,4)
                            if text_w <= image_w - 2*margin and text_h+baseline <= image_h - 2*margin:
                                break
                            font_scale *= 0.9
                        else:
                            raise ValueError('图片太小，无法完整显示标签')
                        text_x = max(margin,min(x1,image_w-margin-text_w))
                        if y1 >= text_h + baseline +margin:
                            text_y = y1- baseline -margin
                        else:
                            text_y = y1 + margin + text_h
                        text_y = max(margin+text_h,min(text_y,image_h-margin-baseline))
                        cv.putText(rectangle,text_sign,(text_x,text_y),font,font_scale,(47,0,0),4,cv.LINE_AA)
                        person_text_positions[check['instance_id']] =(text_y,text_h,baseline)
                    if check['detection_score'] >= min_score and check['label'] == 'head':
                        image_h,image_w = img.shape[:2]
                        margin = 4
                        font_scale = 2
                        x1,y1,x2,y2 = map(int,check['box_xyxy'])
                        left_up_position = tuple((x1,y1))
                        right_up_position = tuple((x2,y2))
                        rectangle = cv.rectangle(img,left_up_position,right_up_position,(255,0,0),5)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        text_sign = check['label']+' '+'#'+str(check['instance_id'])+' '+str(check['detection_score'])
                        for _ in range(60):
                            (text_w,text_h),baseline = cv.getTextSize(text_sign,font,font_scale,4)
                            if text_w <= image_w - 2*margin and text_h+baseline <= image_h - 2*margin:
                                break
                            font_scale *= 0.9
                        else:
                            raise ValueError('图片太小，无法完整显示标签')
                        text_x = max(margin,min(x1,image_w-margin-text_w))
                        if y1 >= text_h + baseline +margin:
                            text_y = y1- baseline -margin
                        else:
                            text_y = y1 + margin + text_h
                        text_y = max(margin+text_h,min(text_y,image_h-margin-baseline))            
                        cv.putText(rectangle,text_sign,(text_x,text_y),font,font_scale,(0,255,0),4,cv.LINE_AA)
                    if check['detection_score'] >= min_score and check['label'] == 'helmet':
                        image_h,image_w = img.shape[:2]
                        margin = 4
                        font_scale = 2
                        x1,y1,x2,y2 = map(int,check['box_xyxy'])
                        left_up_position = tuple((x1,y1))
                        right_up_position = tuple((x2,y2))
                        rectangle = cv.rectangle(img,left_up_position,right_up_position,(255,0,0),5)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        text_sign = check['label']+' '+'#'+str(check['instance_id'])+' '+str(check['detection_score'])
                        for _ in range(60):
                            (text_w,text_h),baseline = cv.getTextSize(text_sign,font,font_scale,4)
                            if text_w <= image_w - 2*margin and text_h+baseline <= image_h - 2*margin:
                                break
                            font_scale *= 0.9
                        else:
                            raise ValueError('图片太小，无法完整显示标签')
                        text_x = max(margin,min(x1,image_w-margin-text_w))
                        if y1 >= text_h + baseline +margin:
                            text_y = y1- baseline -margin
                        else:
                            text_y = y1 + margin + text_h
                        text_y = max(margin+text_h,min(text_y,image_h-margin-baseline))
                        cv.putText(rectangle,text_sign,(text_x,text_y),font,font_scale,(0,0,255),4,cv.LINE_AA)
                instances_by_id = {dic['instance_id']:dic
                    for dic in data['instances']}
                ## 复查处理
                for check_person in report['reviews']:
                    if check_person['image'] == data['image']:
                        normal_y,normal_h,normal_baseline = person_text_positions[check_person['person_id']]
                        position = instances_by_id[check_person['person_id']]['box_xyxy']
                        review_img = img
                        review_img_h,review_img_w = review_img.shape[:2]
                        font_scale_review = 2
                        x1,y1,x2,y2 = map(int,position)
                        left_limit= (x1,y1)
                        right_limit = (x2,y2)
                        review_sign_person = cv.rectangle(review_img,left_limit,right_limit,(0,0,255),5)
                        review_text = 'REVIEW'+':'+check_person['reason']
                        font = cv.FONT_HERSHEY_COMPLEX
                        for _ in range(60):
                            (text_w2,text_h2),baseline = cv.getTextSize(review_text,font,font_scale_review,4)
                            if text_w2 <= review_img_w - 2*margin and text_h2+baseline <= review_img_h - 2*margin:
                                break
                            font_scale_review*=0.9
                        else:
                            raise ValueError('图片太小，无法显示')
                        review_text_x = max(margin,min(x1,review_img_w-margin-text_w2))
                        ## 添加标记，防止重叠(针对纵坐标)
                        normal_top = normal_y-normal_h
                        normal_bottom = normal_y+normal_baseline
                        for _ in range(60):
                            (text_w2,text_h2),baseline = cv.getTextSize(review_text,font,font_scale_review,4)
                            review_text_y = normal_bottom+margin+text_h2
                            if review_text_y<=review_img_h-margin-baseline:
                                break
                            else:
                                review_text_y = normal_top-margin-baseline
                                if review_text_y>=margin+text_h2:
                                    break
                                else:
                                    font_scale_review*=0.9
                        else:
                            raise ValueError('图片太小了，无法显示')
                        cv.putText(review_sign_person,review_text,(review_text_x,review_text_y),
                                font,
                                font_scale_review,
                                (0,144,0),4
                                )       
                    else:
                        continue
            output_image_path = os.path.join('output','images',data['image'])
            success = cv.imwrite(output_image_path,img)
            if success is False:
                raise OSError('图片保存失败')
            return True
total_reports = {}
counts ={}
counts['person'] = 0
counts['head'] = 0
counts['helmet']=0
review_images = []
reviews = []
errors = []
prediction_dir = os.path.join('input','predictions')
for filename in sorted(os.listdir(prediction_dir)):
    if not filename.lower().endswith('.json'):
        continue
    input_text_path = os.path.join('input','predictions',filename)
    if os.path.isfile(input_text_path) is False:
        continue
    report_results = text_process(input_text_path)
    if report_results['errors']:
        errors.extend(report_results['errors'])
        continue
    else:
        image_results = sign_picture(input_text_path,report=report_results)
        if image_results is True:
            counts['person']+=report_results['counts']['person']
            counts['head']+=report_results['counts']['head']
            counts['helmet']+=report_results['counts']['helmet']
            reviews.extend(report_results['reviews'])
            if report_results['review_images']:
                img = report_results['review_images'][0]
                if img not in review_images:
                    review_images.append(img)
        else:
            errors.extend(report_results['errors'])
total_reports['counts'] = counts
total_reports['review_images'] = review_images
total_reports['reviews'] = reviews
total_reports['errors'] = errors
os.makedirs('output',exist_ok=True)
with open('output/report.json','w',encoding='utf-8') as f:
    json.dump(total_reports,f,ensure_ascii=False,indent=2)

# PPE-Grounded-SAM2/
# │
# ├─ main.py                         # 主项目入口
# ├─ association_practice.py         # 你手写的关联算法
# ├─ PPE_check_results.py            # 你正在写的可视化与复查工具
# │
# ├─ input/                          # 复查工具的输入
# │  ├─ images/
# │  │  ├─ a.jpg                     # 原始图片
# │  │  └─ b.jpg
# │  │
# │  └─ predictions/
# │     ├─ a.json                    # a.jpg 的检测与关联数据
# │     └─ b.json
# │
# ├─ output/                         # 复查工具的最终输出
# │  ├─ images/
# │  │  ├─ a.jpg                     # 同名、同尺寸的标记结果图
# │  │  └─ b.jpg
# │  │
# │  └─ report.json                  # 所有图片的汇总报告
# │
# └─ outputs/                        # 主模型推理结果
#    └─ inference/
#       ├─ a/
#       │  ├─ prediction.json
#       │  ├─ overlay.jpg
#       │  └─ masks/
#       └─ b/
#          └─ ...
# data = {
#   "image": "a.jpg", 
#   "width": 640,
#   "height": 480,
#   "instances": [
#     {
#       "instance_id": 1,
#       "label": "person",
#       "detection_score": 0.92,
#       "box_xyxy": [50, 30, 200, 350]
#     },
#     {
#       "instance_id": 2,
#       "label": "head",
#       "detection_score": 0.81,
#       "box_xyxy": [90, 35, 140, 85]
#     }
#   ],
#   "associations": [
#     {
#       "person_id": 1,
#       "head_id": 2,
#       "helmet_id": null
#     }
#   ]
# }              
# report = {
#     "counts": {
#         "person": 1,
#         "head": 1,
#         "helmet": 0
#     },

#     "review_images": ["a.jpg"],

#     "reviews": [
#         {
#             "image": "a.jpg",
#             "person_id": 1,
#             "reason": "helmet_unmatched",
#             "review_status": "pending"
#         }

#     ],
#     "errors": []
# }