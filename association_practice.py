"""PPE 空间关联算法的分块手写练习。"""

from src.schema import BoxXYXY


# 第 1 块：根据检测框的四个边界坐标，计算中心点。
# 只需修改这个函数的函数体；完成后告诉我，我们再进入下一块。
def box_center(box: BoxXYXY) -> tuple[float, float]:
    """返回 (中心 x, 中心 y)。"""
    # TODO: 使用 box.x1、box.y1、box.x2、box.y2 计算中心点。
    centerx = (box.x1 + box.x2)/2  
    centery = (box.y1 + box.y2)/2
    return (centerx,centery)


# 第 1 块（续）：计算检测框面积，后续判断两个框的重叠程度会用到。
def box_area(box: BoxXYXY) -> float:
    """返回检测框的面积。"""
    # TODO: 宽 = x2 - x1；高 = y2 - y1；面积 = 宽 * 高。
    width =  box.x2 - box.x1
    height = box.y2 - box.y1
    area = width * height 
    return area
# 第 2 块：判断一个点是否位于检测框内。
# 本练习约定：左边界、上边界算框内；右边界、下边界算框外。
def point_in_box(x: float, y: float, box: BoxXYXY) -> bool:
    """点在框内时返回 True，否则返回 False。"""
    # TODO: 同时比较 x 与左右边界、y 与上下边界。
    if x >=  box.x1 and x <  box.x2:
        if y>=  box.y1 and y <  box.y2:
            return True
        else:
            return False
    else:
        return False
# 第 3 块：两个检测框的相交面积（请自己写函数头和函数体）。
# 输入：两个 BoxXYXY 检测框。
# 输出：两个框重叠区域的面积；如果完全不重叠，返回 0。
def box_intersection_area(box1:BoxXYXY,box2:BoxXYXY) -> int:
    left = max(box1.x1,box2.x1)
    right = min(box1.x2,box2.x2)
    top = min(box1.y2,box2.y2)
    bottom = max(box1.y1,box2.y1)
    if left < right and bottom < top:
        s = abs(right - left) * abs(top-bottom)
        return s
    else:
        return 0

# 第 4 块：计算两个检测框的交并比 IoU（请自己写函数头和函数体）。
# 输入：两个 BoxXYXY 检测框。
# 输出：0.0 到 1.0 之间的数；越大表示两个框重叠程度越高。
# 交集面积：可调用你写好的 box_intersection_area。
# 并集面积：框 1 面积 + 框 2 面积 - 交集面积。
# IoU：交集面积 / 并集面积。

def box_iou(box1:BoxXYXY,box2:BoxXYXY) -> float:
    s1 = box_area(box1)
    s2 = box_area(box2)
    intersection = box_intersection_area(box1,box2)
    combine = s1 + s2 - intersection
    IoU = intersection / combine
    return IoU
# 第 5 块：人头关联的第一步（在本文件末尾手写）
# 函数名：head_coverage_by_person
# 输入：一个 person 的 BoxXYXY 框、一个 head 的 BoxXYXY 框，顺序如上。
# 输出：相交面积 / 头框面积，结果在 0.0 到 1.0 之间。
def head_coverage_by_person(person_box:BoxXYXY,head_box:BoxXYXY) -> float:
    head_coverage = box_intersection_area(person_box,head_box)/box_area(head_box)
    return head_coverage


# 第 6 块：判断一个 head 框能否与一个 person 框组成候选配对。
# 函数名：head_matches_person（函数头和函数体都由你自己写）
# 输入：person_box、head_box，都是 BoxXYXY。
# 输出：布尔值。下面两个条件必须同时满足，才返回 True：
#   1. 头框落在人框内的比例 >= 0.5（复用 head_coverage_by_person）。
#   2. 头框的中心点在人框内，而且位于人框的上半部分。
#      人框的竖直中线是 (person_box.y1 + person_box.y2) / 2；
#      中心点恰好落在竖直中线上时，不算上半部分。
def head_matches_person(person_box:BoxXYXY,head_box:BoxXYXY)-> bool:
    if head_coverage_by_person(person_box,head_box) >= 0.5:
        s= box_center(head_box)
        check_line = (person_box.y1 + person_box.y2)/2
        if point_in_box(s[0],s[1],person_box):
            if s[1]< check_line:
                return True
            else:
                return False
        else:
            return False
    else:
        return False
# 第 7 块：在多个人框中，为一个头框选择最合适的人框。
# 函数名：best_person_for_head（函数头和函数体都由你自己写）
# 输入：person_boxes（BoxXYXY 列表）和 head_box（一个 BoxXYXY）。
# 输出：选中的人框在 person_boxes 中的索引，从 0 开始；没有候选人时返回 None。
# 规则：
#   1. 只考虑 head_matches_person 返回 True 的人框。
#   2. 在候选人中选 head_coverage_by_person 比例最高的。
#   3. 比例相同时，选原列表中更靠前的那个。
#   4. 不修改输入列表。
# 示例：
#   people = [BoxXYXY(0, 0, 100, 200), BoxXYXY(50, 0, 150, 200)]
#   head = BoxXYXY(70, 10, 110, 50)  -> 1（覆盖比例分别是 0.75、1.0）
#   head = BoxXYXY(20, 10, 60, 50)   -> 0
#   head = BoxXYXY(200, 10, 240, 50) -> None
#   空的 person_boxes 也返回 None。
# 提示：遍历时需要同时拿到索引和人框；先保存当前最佳索引和最佳比例。
def best_person_for_head(person_boxes:list[BoxXYXY],head_box:BoxXYXY) -> int|None:
    person_boxes_copy = person_boxes[:] 
    check = []
    if person_boxes:
        for i,person_box in enumerate(person_boxes_copy):
            if head_matches_person(person_box,head_box):
                ratio = head_coverage_by_person(person_box,head_box)
                check.append((i,ratio))
            else:
                continue
        if check:
            check.sort(key= lambda x:x[1],reverse=True)
            return check[0][0]
        else:
            return None
    else:
        return None
# 第 8 块：判断一个 helmet 框能否与一个 head 框组成候选配对。
# 函数名：helmet_matches_head（函数头和函数体都由你自己写）
# 输入：head_box、helmet_box，都是 BoxXYXY；先传头框，再传头盔框。
# 输出：布尔值。下面两个条件必须同时满足，才返回 True：
#   1. 两框相交面积 / 头盔框面积 >= 0.5。
#   2. 头盔框中心的 y 坐标 <= 头框中心的 y 坐标（头盔位于头的上方）。
# 请复用 box_intersection_area、box_area、box_center。
# 例：head = BoxXYXY(20, 20, 60, 80)
#     helmet = BoxXYXY(25, 15, 55, 35)  -> True（允许头盔一部分高于头框）
#     helmet = BoxXYXY(25, 60, 55, 80)  -> False（在头的下方）
#     helmet = BoxXYXY(70, 15, 100, 35) -> False（没有重叠）
# 这只是第一版空间规则；以后会用真实检测结果检查和调整阈值。
def helmet_matches_head(head_box:BoxXYXY,helmet_box:BoxXYXY) -> bool:
    if box_intersection_area(head_box,helmet_box)/box_area(helmet_box) >=0.5:
        head_box_center_point = box_center(head_box)
        helmet_box_center_point=box_center(helmet_box)
        if head_box_center_point[1] >= helmet_box_center_point[1]:
            return True
        else:
            return False
    return False
# 第 9 块：在多个头框中，为一个头盔框选择最合适的头框。
# 函数名：best_head_for_helmet（函数头和函数体都由你自己写）
# 输入：head_boxes（BoxXYXY 列表）和 helmet_box（一个 BoxXYXY）。
# 输出：选中的头框在 head_boxes 中的索引，从 0 开始；没有候选头时返回 None。
# 规则：
#   1. 只考虑 helmet_matches_head 返回 True 的头框。
#   2. 在候选头中选“相交面积 / 头盔面积”比例最高的。
#   3. 比例相同时，选列表中更靠前的头框。
#   4. 不修改输入列表。
# 示例：
#   heads = [BoxXYXY(20, 20, 60, 80), BoxXYXY(50, 20, 90, 80)]
#   helmet = BoxXYXY(55, 15, 75, 35)   -> 1
#   helmet = BoxXYXY(25, 15, 55, 35)   -> 0
#   helmet = BoxXYXY(120, 15, 150, 35) -> None
#   空的 head_boxes 也返回 None。
# 提示：可以借鉴第 7 块，但这里比较的是“头盔落在头框内”的比例。
def best_head_for_helmet(head_boxes:list[BoxXYXY],helmet_box:BoxXYXY)-> int|None:
    if head_boxes:
        results = []
        copy_boxes = head_boxes[:]
        for index,position in enumerate(copy_boxes):
            if helmet_matches_head(position,helmet_box) == True:
                area = box_area(helmet_box)
                ratio = box_intersection_area(position,helmet_box)/area
                results.append((index,area,ratio))
        if results:
            results.sort(key = lambda x:x[2],reverse=True)
            return results[0][0]
        else:
            return None
    else:
        return None


# 第 10 块 A：收集整张图的所有有效“人—头”候选，为后续处理配对冲突做准备。
# 函数名：person_head_candidates（请自己写函数头和实现）
# 输入：person_boxes（人框列表）、head_boxes（头框列表），元素都是 BoxXYXY。
# 输出：[(人框原始索引, 头框原始索引, 头框落在人框内的比例), ...]。
# 两种索引都从 0 开始。
# 规则：
#   1. 检查每个人框与每个头框的组合。
#   2. 只保留 head_matches_person 返回 True 的组合。
#   3. 比例复用 head_coverage_by_person 计算。
#   4. 按比例降序；同分时按头框索引升序，再按人框索引升序。
#   5. 任一输入列表为空或没有有效组合时，返回 []；不修改输入列表。
#
# 示例：
# people = [BoxXYXY(0, 0, 100, 200), BoxXYXY(50, 0, 150, 200)]
# heads = [BoxXYXY(70, 10, 110, 50), BoxXYXY(20, 10, 60, 50)]
# 预期返回：[(1, 0, 1.0), (0, 1, 1.0), (0, 0, 0.75)]
#
# 注意：这一步保留所有候选，不要直接调用 best_person_for_head 只选一个。
# 原因：最佳候选如果被占用，后续还可能需要考虑次优候选。
# 下一小步才从这些候选中选择互不冲突的配对。
def person_head_candidates(person_boxes:list[BoxXYXY],head_boxes:list[BoxXYXY]) -> list:
    copy_person_boxes = person_boxes[:]
    copy_head_boxes = head_boxes[:]
    if copy_person_boxes and copy_head_boxes:
        results = []
        for person_index,person_position in enumerate(copy_person_boxes):
            for head_index,head_position in enumerate(copy_head_boxes):
                if head_matches_person(person_position,head_position) == True:
                    IOU = head_coverage_by_person(person_position,head_position)
                    results.append((person_index,head_index,IOU))
        if results:
            results.sort(key=lambda x:(-x[2],x[1],x[0]))
            return results
        else:
            return []
    else:
        return []


# 第 10 块 B：从排好序的人—头候选中选择互不冲突的配对。
# 函数名：select_person_head_matches（请自己写函数头和实现）
# 输入：candidates，即第 10 块 A 返回的、已按优先级排好序的三元组列表。
# 每项格式：(人索引, 头索引, 覆盖比例)。无需再次排序。
# 输出：被选中的三元组列表，顺序与它们在 candidates 中出现的顺序一致。
# 规则：
#   1. 从前往后逐项检查候选。
#   2. 仅当该人索引和头索引都没有被已选配对使用时，才保留这一项。
#   3. 保留后，记录该人和该头已被占用。
#   4. 任意一方已被占用，就跳过这一项并继续检查后面的候选。
#   5. 空输入返回 []；不修改输入列表。
# 示例 1：
# [(1, 0, 1.0), (0, 1, 1.0), (0, 0, 0.75)]
# -> [(1, 0, 1.0), (0, 1, 1.0)]
# 示例 2：
# [(0, 0, 0.95), (1, 0, 0.90), (0, 1, 0.85), (1, 1, 0.80)]
# -> [(0, 0, 0.95), (1, 1, 0.80)]
# 提示：可以用两个集合，分别记录已经使用的人索引和头索引。
# 这是高分优先的贪心选择，不保证配对数量或总分的全局最优。
def select_person_head_matches(candidates) -> list:
    copy_candidates = candidates[:]
    results = []
    used_people= []
    used_head = []
    if candidates:
        for person_index,head_index,IOU in copy_candidates:
            if person_index not in used_people:
                if head_index not in used_head:
                    results.append((person_index,head_index,IOU))
                    used_people.append(person_index)
                    used_head.append(head_index)
                else:
                    continue
            else:
                continue
        return results
    else:
        return []
# 第 11 块 A：收集所有有效“头—头盔”候选。
# 函数名：head_helmet_candidates（请自己写函数头和实现）
# 输入：head_boxes（头框列表）、helmet_boxes（头盔框列表），元素都是 BoxXYXY。
# 输出：[(头框原始索引, 头盔框原始索引, 覆盖比例), ...]，索引从 0 开始。
# 规则：
#   1. 检查每个头框与每个头盔框的组合。
#   2. 只保留 helmet_matches_head 返回 True 的组合。
#   3. 覆盖比例 = 两框相交面积 / 头盔框面积。
#   4. 按覆盖比例降序；同分时按头盔索引升序，再按头索引升序。
#   5. 任一输入列表为空或没有候选时返回 []；不修改输入列表。
#   6. 保留所有有效候选，不要直接用 best_head_for_helmet 只选一个。
# 示例：
# heads = [BoxXYXY(20, 20, 60, 80), BoxXYXY(40, 20, 80, 80)]
# helmets = [BoxXYXY(45, 20, 65, 40), BoxXYXY(25, 15, 55, 35)]
# 预期返回：[(1, 0, 1.0), (0, 0, 0.75), (0, 1, 0.75)]
# 第 11 块 B 再从这些候选中选择一对一配对。
def head_helmet_candidates(head_boxes:list[BoxXYXY],helmet_boxes:list[BoxXYXY]) -> list:
    if head_boxes and helmet_boxes:
        results = []
        copy_head_boxes = head_boxes[:]
        copy_helmet_boxes = helmet_boxes[:]
        for head_index,head_check in enumerate(copy_head_boxes):
            for helmet_index,helmet_check in enumerate(copy_helmet_boxes):
                if helmet_matches_head(head_check,helmet_check) ==True:
                    cover_ratio = box_intersection_area(head_check,helmet_check)/box_area(helmet_check)
                    results.append((head_index,helmet_index,cover_ratio))
        results.sort(key=lambda x:(-x[2],x[1],x[0]))            
        return results
    else:
        return []
# 第 11 块 B：选择互不冲突的“头—头盔”配对。
# 函数名：select_head_helmet_matches（请自己写函数头和实现）
# 输入：candidates，第 11 块 A 返回的已排序列表。
# 每项：(头索引, 头盔索引, 覆盖比例)。无需再排序。
# 输出：选中的三元组列表，保持它们在 candidates 中的先后顺序。
# 规则：
#   1. 从前往后检查候选。
#   2. 头和头盔都未被已选配对使用时，保留该候选，并记录两方已占用。
#   3. 任意一方已被占用，就跳过，继续检查下一项。
#   4. 每个头最多配一个头盔，每个头盔最多配一个头。
#   5. 空输入返回 []；不修改输入列表。
# 示例 1：
# [(1, 0, 1.0), (0, 0, 0.75), (0, 1, 0.75)]
# -> [(1, 0, 1.0), (0, 1, 0.75)]
# 示例 2：
# [(0, 0, 0.95), (1, 0, 0.90), (0, 1, 0.85), (1, 1, 0.80)]
# -> [(0, 0, 0.95), (1, 1, 0.80)]
# 两类索引要分别记录：头0和头盔0是不同对象，不能共用一个占用集合。
def select_head_helmet_matches(candidates:list)->list:
    results = []
    used_heads = set()
    used_helmet = set()
    for select in candidates:
        if select[0] not in used_heads and select[1] not in used_helmet:
            results.append((select[0],select[1],select[2]))
            used_heads.add(select[0])
            used_helmet.add(select[1])
        else:
            continue
    return results
# 第 12 块：把两组配对汇总为每个人的关联结果。
# 本块只写一个函数，可以分两步完成：
#   第一步：整理“人索引 -> 头索引”“头索引 -> 头盔索引”两张查找表。
#   第二步：按人索引顺序，为每个人生成一条结果，并判断复查原因。
# 函数名：summarize_person_associations（函数头和实现都由你自己写）
# 输入，顺序如下：
#   1. person_boxes：人框列表，元素为 BoxXYXY。
#   2. person_head_matches：第 10 块 B 选中的配对列表。
#      每项为 (人索引, 头索引, 覆盖比例)。
#   3. head_helmet_matches：第 11 块 B 选中的配对列表。
#      每项为 (头索引, 头盔索引, 覆盖比例)。
# 约定：两组配对已经合法且互不冲突，直接使用，不重新排序或计算几何关系。
# 输出：字典组成的列表，每个人恰好一条，按 person_index 从小到大排列。
# 每条字典固定包含四个键：
#   person_index：这个人在 person_boxes 中的索引。
#   head_index：配对头的索引；没有配对时为 None。
#   helmet_index：通过这个人的头配对到的头盔索引；没有配对时为 None。
#   review_reason：按下面三种情况取值。
# 规则：
#   1. 人没有配到头：head_index、helmet_index 都为 None，
#      review_reason 为 'head_unmatched'。
#   2. 人配到了头，但这个头没有配到头盔：helmet_index 为 None，
#      review_reason 为 'helmet_unmatched'。
#   3. 人、头、头盔均配对完整：review_reason 为 None。
#      这只表示空间关联完整，不等于已经确认检测正确或佩戴合规。
#   4. 即使某个人完全没有配对，也不能漏掉这个人的结果。
#   5. 没有配给任何人的头，其头盔配对不生成额外结果。
#   6. person_boxes 为空时返回 []；两组配对为空时，所有人都是 head_unmatched。
#   7. 不修改三个输入列表；覆盖比例不需要写入输出字典。
# 注意：索引 0 是有效配对，不要用 if head_index 判断是否有头；应判断是否为 None。
# 这里都是各类框列表中的索引，还不是检测 JSON 中的 instance_id。
# 示例：
# person_boxes = [
#     BoxXYXY(0, 0, 100, 200),
#     BoxXYXY(120, 0, 220, 200),
#     BoxXYXY(240, 0, 340, 200),
# ]
# person_head_matches = [(0, 1, 1.0), (2, 0, 0.8)]
# head_helmet_matches = [(1, 0, 0.9)]
# 预期返回：
# [
#     {'person_index': 0, 'head_index': 1, 'helmet_index': 0,
#      'review_reason': None},
#     {'person_index': 1, 'head_index': None, 'helmet_index': None,
#      'review_reason': 'head_unmatched'},
#     {'person_index': 2, 'head_index': 0, 'helmet_index': None,
#      'review_reason': 'helmet_unmatched'},
# ]
# 提示：先建立两张查找表，再遍历所有人；结果列表应在处理完所有人后返回。
def summarize_person_associations(person_boxes:list[BoxXYXY],person_head_matches:list,head_helmet_matches:list)->list:
    final_results = []
    person_head = {check_person_head[0]:check_person_head[1]
        for check_person_head in person_head_matches}
    head_helmet = {check_head_helmet[0]:check_head_helmet[1]
        for check_head_helmet in head_helmet_matches}
    for index,person in enumerate(person_boxes):
        results_dict = {
            'person_index':index,
            'head_index':None,
            'helmet_index':None,
            'review_reason':None
        }
        if index not in person_head:
            results_dict['review_reason'] = 'head_unmatched'
        else:
            results_dict['head_index'] = person_head[index]
            if person_head[index] not in head_helmet:
                results_dict['review_reason'] = 'helmet_unmatched'
            else:
                results_dict['helmet_index'] = head_helmet[person_head[index]]
        final_results.append(results_dict)
    return final_results
# 第 13 块：整张图的 PPE 空间关联入口。
# 本块只写一个函数，复用前面已经完成的函数。
# 函数名：associate_ppe_boxes（函数头和实现都由你自己写）
# 输入，顺序如下：
#   1. person_boxes：人框列表，元素为 BoxXYXY。
#   2. head_boxes：头框列表，元素为 BoxXYXY。
#   3. helmet_boxes：头盔框列表，元素为 BoxXYXY。
# 输出：第 12 块格式的字典列表，每个人恰好一条，按人索引升序。
# 固定字段：person_index、head_index、helmet_index、review_reason。
# 索引对应各自的原始输入列表；缺失配对用 None。
# 按下面顺序衔接五个已有函数：
#   1. person_head_candidates：用人框和头框生成全部有效人—头候选。
#   2. select_person_head_matches：从上一步候选中选出一对一配对。
#   3. head_helmet_candidates：用全部头框和头盔框生成有效头—头盔候选。
#   4. select_head_helmet_matches：从上一步候选中选出一对一配对。
#   5. summarize_person_associations：传入人框列表及两组已选配对，返回最终结果。
# 每一步的返回值需要由变量接收，再交给后续函数使用。
#
# 规则：
#   1. 复用已有的空间规则、排序规则和贪心选择，不重新实现这些逻辑。
#   2. 第 3 步使用原始 head_boxes 全部头框，不提前过滤或重排，否则索引会变化。
#   3. 没有人时返回 []；有但没配到头的人也必须出现在最终列表里。
#   4. 头框为空时，所有人的 review_reason 都是 'head_unmatched'。
#   5. 头盔框为空时，已配到头的人为 'helmet_unmatched'，没配到头的仍为 'head_unmatched'。
#   6. 不修改三个输入列表，不重新编号，不额外包一层列表或字典。
#   7. 本函数输入是三类框列表，直接返回关联结果；JSON 与图片的读写由其他模块负责。
#
# 示例：
# person_boxes = [
#     BoxXYXY(0, 0, 100, 200),
#     BoxXYXY(120, 0, 220, 200),
#     BoxXYXY(240, 0, 340, 200),
# ]
# head_boxes = [
#     BoxXYXY(260, 10, 300, 50),  # 头索引 0，属于人索引 2。
#     BoxXYXY(20, 10, 60, 50),    # 头索引 1，属于人索引 0。
# ]
# helmet_boxes = [BoxXYXY(25, 10, 55, 30)]
# 预期返回：
# [
#     {'person_index': 0, 'head_index': 1, 'helmet_index': 0,
#      'review_reason': None},
#     {'person_index': 1, 'head_index': None, 'helmet_index': None,
#      'review_reason': 'head_unmatched'},
#     {'person_index': 2, 'head_index': 0, 'helmet_index': None,
#      'review_reason': 'helmet_unmatched'},
# ]
# 三个输入都为空时，预期返回 []。
# 提示：关注每个被调用函数的输入与返回值，最后返回第 12 块生成的结果。
def associate_ppe_boxes(person_boxes:list[BoxXYXY],head_boxes:list[BoxXYXY],helmet_boxes:list[BoxXYXY])->list:
    person_head_candidates_ = person_head_candidates(person_boxes,head_boxes)
    select_person_head_matches_ = select_person_head_matches(person_head_candidates_)
    head_helmet_candidates_= head_helmet_candidates(head_boxes,helmet_boxes)
    select_head_helmet_matches_ = select_head_helmet_matches(head_helmet_candidates_)
    summarize_person_associations_= summarize_person_associations(person_boxes,select_person_head_matches_,select_head_helmet_matches_)
    return summarize_person_associations_

    
