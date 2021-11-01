import cv2
import numpy as np
import random
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
import logging
import json
import os


class DisplayChinese:
    def __init__(self):
        self.font_path = "data/longyin55.ttf"

    def putText(self, img, text, pos, color=(255, 255, 255), textSize=30, align='left'):
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        fontText = ImageFont.truetype(self.font_path, textSize, encoding="utf-8")
        draw.text(pos, text, color, font=fontText, align=align)
        return np.array(img)

class HuangChangTiCou:
    def __init__(self):
        cv2.namedWindow("hctc", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow("room", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.namedWindow("panel", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        # 设置中文界面显示
        self.dc = DisplayChinese()
        # 隐藏地图界面开关
        self.hide = False
        # 出宫模式
        self.out = False
        # 加载头像
        self.head_pic_dict = self.load_head_pic()
        # 初始化地图
        self.map = self.create_map()
        self.room = np.zeros((500, 800, 3), np.uint8)
        self.default_map = self.map.copy()
        # 初始化人物数据
        self.person_info = self.put_person_default()
        # 跟随开关
        self.follow = False
        # 帮助开关
        self.help = False
        self.help_img = cv2.imread('data/keyboard-layout.png')
        # 读取存放地图信息的json文件
        with open('data/map.json', 'r', encoding='utf-8') as f:
            self.map_dict = json.load(f)
        # 随机生成食物分布
        self.random_food()
        # 随机生成弹药分布
        self.random_ammo()
        # 创建房间名称字典，用于自定义人物位置
        self.room_name_dict = {}
        for k, v in self.map_dict.items():
            self.room_name_dict[v['name']] = list(eval(k))
        # 创建人物行进信息栈,用于人物行进撤销操作,每一个元素为一个元组
        self.act_stack = []

        # 创建动作优先级字典
        self.act_dict = self.gen_act_dict()
        self.person_id_dict = {48: '兰', 49: '袁丛', 50: '陈必', 51: '刘钥', 52: '袁星', 53: '柳舒之', 54: '李媛姗', 55: '柳悦己'}
        self.person_state_dict = {'兰': (0, 0), '袁丛': (0, 1), '陈必': (0, 2), '刘钥': (0, 3),
                                  '袁星': (2, 0), '柳舒之': (2, 1), '李媛姗': (2, 2), '柳悦己': (2, 3)}
        # 命卦字典
        self.minggua = {0: '离', 1: '坎', 2: '坤', 3: '震', 4: '巽', 5: '坤艮', 6: '乾', 7: '兑', 8: '艮'}
        # 初始化状态板
        self.update_person_state()
        # 设置自动行进（初始为手动）
        self.auto = False


    def create_map(self):
        """
        创建地图
        :return:
        """
        map = np.zeros((1200, 1200, 3), np.uint8)

        for i in [1, 2, 4, 5]:
            cv2.line(map, (0, i * 200), (1200, i * 200), (255, 255, 255), 3)
            cv2.line(map, (i * 200, 0), (i * 200, 1200), (255, 255, 255), 3)

        return map

    def load_head_pic(self):
        """
        加载头像数据
        :return:
        """
        img_root = 'data/head_pic'
        img_name_list = os.listdir(img_root)
        head_pic_dict = {}
        for img_name in img_name_list:
            img_path = os.path.join(img_root, img_name)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            img = cv2.resize(img, (140, 140))
            person_name = img_name.split('.')[0]
            head_pic_dict[person_name] = img

        return head_pic_dict


    def show_map(self):
        """
        显示地图
        :return:
        """
        cv2.imshow('hctc', self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def init_map(self):
        """
        初始化地图
        :return:
        """
        self.map = self.create_map()
        self.person_info = self.put_person_default()

    def put_person_default(self):
        """
        设置默认的角色信息，对应位置、颜色、行动优先级
        """
        person_info = {}
        person_info['兰'] = {'lc': [2, 6], 'color': (255, 255, 0), 'order': 0, 'healthy': 8, 'chi': 8, 'energy': 9, 'time': (0, 0, 0), 'speed': 1, 'direction': None, 'weapon': ['日94式'], 'item': ['古钱币*3', '刘寄奴', '钩吻', '景天', '王不留行'], 'ability': ['六爻（辨吉凶、分阴阳）', '叶底藏花', '包扎', '远程冷兵加成'], 'gua': '乾', 'blood': 0, 'poision': 0}
        person_info['柳舒之'] = {'lc': [1, 1], 'color': (0, 0, 255), 'order': 5, 'healthy': 7, 'chi': 8, 'energy': 11, 'time': (0, 0, 0), 'speed': 0.5, 'direction': None, 'weapon': ['奇门擎天伞'], 'item': ['急救包（针筒，手术刀）', '小药箱（亚硝酸钠，硫代硫酸钠，氯化钾，肾上腺素，好伤药*2）'], 'ability': ['急救', '注射', '药剂加成'], 'gua': '离', 'blood': 0, 'poision': 0}
        person_info['柳悦己'] = {'lc': [1, 6], 'color': (255, 0, 0), 'order': 7, 'healthy': 6, 'chi': 8, 'energy': 12, 'time': (0, 0, 0), 'speed': 0.5, 'direction': None, 'weapon': ['奇门纳地伞'], 'item': ['《奇门密要》'], 'ability': ['奇门遁甲(排盘，解卦)', '长枪及飞刀加成', '身法'], 'gua': '兑', 'blood': 0, 'poision': 0}
        person_info['刘钥'] = {'lc': [3.5, 1], 'color': (0, 255, 255), 'order': 3, 'healthy': 8, 'chi': 8, 'energy': 9, 'time': (0, 0, 0), 'speed': 0.5, 'direction': None, 'weapon': ['吹管（画笔）', '驳壳枪'], 'item': ['梦蝶香', '忘忧散', '断魂香', '香包*2', '钢丝录音机', '古钱币*3', '手表（坏的）', '清醒药*3', '麻痹散', '魅幽香'], 'ability': ['飞熊入梦', '香散加成'], 'gua': '坎', 'blood': 0, 'poision': 0}
        person_info['袁丛'] = {'lc': [6, 1], 'color': (128, 128, 128), 'order': 1, 'healthy': 9, 'chi': 8, 'energy': 8, 'time': (0, 0, 0), 'speed': 1, 'direction': None, 'weapon': ['相机(含击发器)'], 'item': ['《时闻纪要》', '手表(坏的)', '证件'], 'ability': ['伪装看破', '对答如流', '察言观色', '枪械加成'], 'gua': '坤', 'blood': 0, 'poision': 0}
        person_info['袁星'] = {'lc': [6, 3.5], 'color': (150, 0, 100), 'order': 4, 'healthy': 12, 'chi': 8, 'energy': 5, 'time': (0, 0, 0), 'speed': 1, 'direction': None, 'weapon': ['消音M1911'], 'item': ['金针', '钢丝绳', '《军统电码本》', '手表'], 'ability': ['易容', '神枪手', '全兵器加成'], 'gua': '艮', 'blood': 0, 'poision': 0}
        person_info['陈必'] = {'lc': [1, 3.5], 'color': (0, 128, 255), 'order': 2, 'healthy': 10, 'chi': 8, 'energy': 3, 'time': (0, 0, 0), 'speed': 1, 'direction': None, 'weapon': ['阴阳宣花斧'], 'item': ['探阴爪', '《寻龙笔记》', '风水罗盘'], 'ability': ['耐毒体质', '近战冷兵加成'], 'gua': '震', 'blood': 0, 'poision': 0}
        person_info['李媛姗'] = {'lc': [3.5, 6], 'color': (255, 255, 255), 'order': 6, 'healthy': 6, 'chi': 8, 'energy': 12, 'time': (0, 0, 0), 'speed': 0.5, 'direction': None, 'weapon': ['折叠手枪'], 'item': ['手包', '《军统电码本》', '诗', '手表'], 'ability': ['梅花易数', '巧连神数', 'buff'], 'gua': '巽', 'blood': 0, 'poision': 0}

        for k, v in person_info.items():
            cx = int(200 * (v['lc'][0] - 0.5))
            cy = int(200 * (v['lc'][1] - 0.5))
            cv2.circle(self.map, (cx, cy), 15, v['color'], 30)

        return person_info

    def save_person_state(self):
        """
        实时存储角色信息
        :return:
        """

        with open('data/person_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.person_info, f, ensure_ascii=False, indent=4)

    def load_person_state(self):
        """
        从外部读取人物信息
        :return:
        """
        print ('开始加载人物信息......')
        json_file_path = 'data/person_info.json'
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                self.person_info = json.load(f)
            print ('信息加载更新完毕, 当前人物信息为：')
            print (self.person_info)
        else:
            print('没有找到人物信息文件')

    def random_food(self):
        """
        随机分配食物
        :return:
        """
        for key, value in self.map_dict.items():
            if key == "(1, 5)":
                food_num = 5 + random.choice([0, 0, 1, 1, 1, 2, 2, 3])
            elif key == "(3.5, 3.5)" or key == "(5, 1)":
                food_num = 0
            else:
                food_num = random.choice([0, 0, 1, 1, 1, 2, 2, 3])

            self.map_dict[key]["food"] = food_num

    def random_ammo(self):
        """
        随机分配弹药
        :return:
        """
        ammo_list = ['7.63', '11.43', '6.5', '8', '6.35']
        ammo_type = random.choice(ammo_list)

        for key, value in self.map_dict.items():
            if key == "(6, 3.5)" or key == "(1, 3.5)":
                ammo_num = 7 + random.choice([0, 0, 1, 1, 1, 2, 2, 3])
            elif key == "(3.5, 3.5)":
                ammo_num = 0
            else:
                ammo_num = random.choice([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7])

            self.map_dict[key]["ammo"][ammo_type] = ammo_num

    def random_guizi(self):
        """
        随机分配鬼子
        :return:
        """
        for key, value in self.map_dict.items():
            if key == "(1, 5)":
                guizi_num = 5 + random.choice([0, 0, 1, 1, 1, 2, 2, 3])
            elif key == "(3.5, 3.5)" or key == "(5, 1)":
                guizi_num = 0
            else:
                guizi_num = random.choice([0, 0, 1, 1, 1, 2, 2, 3])

            self.map_dict[key]["guizi"] = guizi_num


    def judge_meet(self):
        """
        判断是否相遇,并更新行动优先级
        """
        name_list = list(self.person_info.keys())

        for name, info in self.person_info.items():
            # 遇到的伙伴列表
            meet_list = []
            for person in name_list:
                # 如果是自己则跳过
                if name == person:
                    meet_list.append(name)
                    continue
                # 如果位置相同且优先级不同则说明是第一次遇上，或者为非自动模式
                if self.person_info[name]['lc'] == self.person_info[person]['lc']:
                    if self.person_info[name]['order'] != self.person_info[person]['order']:
                        meet_list.append(person)
            # 如果遇到了人，且当前为自动模式，则同组的所有人更新优先级为组内最靠前的优先级
            if self.auto:
                min_priority = min([self.person_info[name]['order'] for name in meet_list])
                for name in meet_list:
                    self.person_info[name]['order'] = min_priority
            # 同步为最慢的时间
            max_time = 0
            for name in meet_list:
                time_tuple = self.person_info[name]['time']
                time = time_tuple[0] * 24 * 60 + time_tuple[1] * 60 + time_tuple[2]
                if time > max_time:
                    max_time = time

            d = int(max_time // (60 * 24))
            h = int(max_time // 60 - 24 * d)
            m = int(max_time % 60)

            for name in meet_list:
                self.person_info[name]['time'] = (d, h, m)


    def draw_state(self, person, healthy, chi, energy):
        """
        绘制当前角色的状态
        :param person: 人物名
        :param healthy: 精血
        :param chi: 气脉
        :param energy: 神识
        :return:
        """
        # 定义单位血条大小
        rw = 30
        rh = 50
        # 定义血条间隔
        x_gap = 10
        y_gap = 10
        x1, y1 = self.person_state_dict[person]
        # 定义初始位置
        x1 = x1 * 400
        y1 = y1 * 200
        # 定义头像位置
        bar_left = 300
        head_left = 80
        head_size = 70
        # 定义文字大小
        font_size = 35

        # 放置头像
        head_img = self.head_pic_dict[person]
        self.panel[y1 + 20:y1 + 20 + 2 * head_size, x1 + head_left:x1 + head_left + 2 * head_size, :] = head_img
        self.panel = self.dc.putText(self.panel, person, (x1 + head_left, y1 + 20 + 2 * head_size), (255, 255, 255), textSize=font_size)
        # 放置状态
        if self.person_info[person]['poision']:
            self.panel = self.dc.putText(self.panel, '毒', (x1 + head_left + 160, y1 - 70 + 2 * head_size), (0, 255, 0), textSize=font_size)
        if self.person_info[person]['blood']:
            self.panel = self.dc.putText(self.panel, '血', (x1 + head_left + 160, y1 - 130 + 2 * head_size), (0, 0, 255), textSize=font_size)
        # 更新血量
        for i in range(healthy):
            cv2.rectangle(self.panel, (x1 + bar_left + (rw + x_gap) * i, y1 + y_gap),
                          (x1 + bar_left + rw + (rw + x_gap) * i, y1 + y_gap + rh), (0, 0, 255), -1)
        for j in range(chi):
            cv2.rectangle(self.panel, (x1 + bar_left + (rw + x_gap) * j, y1 + 2 * y_gap + rh),
                          (x1 + bar_left + rw + (rw + x_gap) * j, y1 + 2 * y_gap + 2 * rh), (0, 255, 0), -1)
        for k in range(energy):
            cv2.rectangle(self.panel, (x1 + bar_left + (rw + x_gap) * k, y1 + 3 * y_gap + 2 * rh),
                          (x1 + bar_left + rw + (rw + x_gap) * k, y1 + 3 * y_gap + 3 * rh), (255, 255, 0), -1)

    def update_person_state(self):
        """
        更新状态板
        :return:
        """
        self.panel = np.zeros((800, 1600, 3), np.uint8)
        for person in self.person_state_dict.keys():
            healthy = self.person_info[person]['healthy']
            chi = self.person_info[person]['chi']
            energy = self.person_info[person]['energy']
            self.draw_state(person, healthy, chi, energy)

    def update_person_info(self):
        """
        在地图上更新人物的位置信息
        :return:
        """
        self.map = self.default_map.copy()
        for k, v in self.person_info.items():
            cx = int(200 * (v['lc'][0] - 0.5))
            cy = int(200 * (v['lc'][1] - 0.5))
            cv2.circle(self.map, (cx, cy), 15, v['color'], 30)

    def move_choice(self, lc):
        """
        根据人物所在房间给出可以移动的方向
        :param lc: 当前人物所在房间位置坐标
        :return:
        """
        x, y = lc
        # 在第一列
        if x == 1:
            if y in [1, 2, 3.5]:
                return ['右', '下']
            elif y == 5:
                return ['右']
            elif y == 6:
                return ['右', '上']
        # 在第二列
        elif x == 2:
            if y == 1:
                return ['下']
            elif y == 2:
                return ['右', '下']
            elif y == 3.5:
                return ['下']
            elif y == 5:
                return ['右']
            elif y == 6:
                return ['上']

        # 在第三、四列
        elif x == 3.5:
            if y in [1, 6]:
                return ['左', '右']
            elif y == 2:
                return ['上', '下']
            elif y == 3.5:
                return []
            elif y == 5:
                return ['右', '下']

        # 在第五列
        elif x == 5:
            if y == 1:
                return ['下']
            elif y == 2:
                return ['左']
            elif y == 3.5:
                return ['上']
            elif y == 5:
                return ['上']
            elif y == 6:
                return ['上', '右']

        # 在第六列
        elif x == 6:
            if y == 1:
                return ['左', '下']
            elif y == 2:
                return ['左']
            elif y in [3.5, 5]:
                return ['左', '上']
            elif y in [6]:
                return ['上']

        # 如果出现异常
        print (x, y)
        cv2.imshow('map', self.map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def random_move(self, lc):
        """
        根据当前人物位置进行随机移动
        :param lc: 当前人物所在房间位置坐标
        :return:
        """
        mc = self.move_choice(lc)

        if mc == []:
            return lc, '到达梓宫'
        else:
            direction = random.choice(mc)

        x, y = lc
        if direction == '左':
            if x == 3.5:
                x -= 1.5
            else:
                x -= 1
        elif direction == '右':
            if x == 3.5:
                x += 1.5
            else:
                x += 1
        elif direction == '上':
            if y == 3.5:
                y -= 1.5
            else:
                y -= 1
        elif direction == '下':
            if y == 3.5:
                y += 1.5
            else:
                y += 1

        if x in [3, 4]:
            x = 3.5

        if y in [3, 4]:
            y = 3.5

        x = x if x == 3.5 else int(x)
        y = y if y == 3.5 else int(y)

        return [x, y], direction

    def move(self, lc, direction):

        x, y = lc
        if direction == '左':
            if x == 3.5:
                x -= 1.5
            else:
                x -= 1
        elif direction == '右':
            if x == 3.5:
                x += 1.5
            else:
                x += 1
        elif direction == '上':
            if y == 3.5:
                y -= 1.5
            else:
                y -= 1
        elif direction == '下':
            if y == 3.5:
                y += 1.5
            else:
                y += 1

        if x in [3, 4]:
            x = 3.5

        if y in [3, 4]:
            y = 3.5

        x = x if x == 3.5 else int(x)
        y = y if y == 3.5 else int(y)

        return [x, y], direction

    def hourglass(self, person_name):
        """
        根据人物速度进行该单位的时间推移
        :param person_name: 当前人物名
        :return: (天, 时, 分)
        """

        speed = self.person_info[person_name]['speed']
        time_tuple = self.person_info[person_name]['time']
        # 更新时间
        time = time_tuple[0] * 24 * 60 + time_tuple[1] * 60 + time_tuple[2] + 30 / speed
        d = int(time // (60 * 24))
        h = int(time // 60 - 24 * d)
        m = int(time % 60)
        self.person_info[person_name]['time'] = (d, h, m)

        return (d, h, m)


    def gen_act_dict(self):
        """
        生成行动优先级字典
        """
        act_dict = OrderedDict()
        for name, v in self.person_info.items():
            order = self.person_info[name]['order']
            if order not in list(act_dict.keys()):
                act_dict[order] = [name]
            else:
                act_dict[order].append(name)
        act_dict = sorted(act_dict.items(), key=lambda x: x[0])

        return act_dict

    def make_room(self, person_name):
        """
        展示坐标对应的房间
        """
        lc = self.person_info[person_name]['lc']
        person_color = self.person_info[person_name]['color']
        healthy = self.person_info[person_name]['healthy']
        chi = self.person_info[person_name]['chi']
        energy = self.person_info[person_name]['energy']
        blood = self.person_info[person_name]['blood']
        poision = self.person_info[person_name]['poision']
        index = str(tuple(lc))
        room = np.zeros((500, 800, 3), np.uint8)

        # 从外部加载的地图字典信息
        room_name = self.map_dict[index]["name"]
        room_color = self.map_dict[index]["color"]
        room_food = self.map_dict[index]["food"]
        room_guizi = self.map_dict[index]["guizi"]
        ammo_type, ammo_num = list(self.map_dict[index]["ammo"].items())[0]
        room_infos = self.map_dict[index]["info"]
        room_out = self.map_dict[index]["out"]
        room_in = self.map_dict[index]["in"]
        cv2.rectangle(room, (50, 50), (450, 450), room_color, 3)
        # 添加房间名
        # cv2.putText(room, room_name, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        room = self.dc.putText(room, room_name, (50, 10), (255, 255, 255))
        # 添加当前人物姓名
        room = self.dc.putText(room, person_name, (350, 10), (255, 255, 255))
        # 添加鬼子数量
        if self.out:
            room = self.dc.putText(room, f'{room_guizi}个鬼子', (315, 50), (255, 255, 255))
        # 添加状态
        cv2.circle(room, (550, 50), 15, person_color, 30)
        room = self.dc.putText(room, f'精血:  {healthy}', (600, 0), (255, 255, 255))
        room = self.dc.putText(room, f'气脉:  {chi}', (600, 30), (255, 255, 255))
        room = self.dc.putText(room, f'神识:  {energy}', (600, 60), (255, 255, 255))
        if poision:
            room = self.dc.putText(room, '毒', (720, 30), (0, 255, 0))
        if blood:
            room = self.dc.putText(room, '血', (750, 30), (0, 0, 255))
        room = self.dc.putText(room, f'气脉:  {chi}', (600, 30), (255, 255, 255))
        # 添加时间
        day, hour, minute = self.person_info[person_name]['time']
        room = self.dc.putText(room, f'时间: {day}天{hour}时{minute}分', (550, 100), (255, 255, 255))

        line = 1

        # 添加食物信息
        if room_food != 0:
            room = self.dc.putText(room, f'{room_food}包食物', (550, 75 + 75 * line), (255, 255, 255), textSize=50, align='center')
            line += 1
        # 添加弹药信息
        if ammo_num != 0:
            room = self.dc.putText(room, f'{ammo_num}发{ammo_type}mm', (550, 75 + 75 * line), (255, 255, 255), textSize=50, align='center')
            line += 1
        # 添加道具信息
        for room_info in room_infos:
            room = self.dc.putText(room, room_info, (550, 75 + 75 * line), (255, 255, 255), textSize=50, align='center')
            line += 1
        # 添加出口信息
        if len(room_out) > 0:
            for dir in room_out:
                if dir == '左':
                    cv2.arrowedLine(room, (160, 250), (60, 250), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '右':
                    cv2.arrowedLine(room, (340, 250), (440, 250), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '上':
                    cv2.arrowedLine(room, (250, 160), (250, 60), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '下':
                    cv2.arrowedLine(room, (250, 340), (250, 440), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        # 添加入口信息
        if len(room_in) > 0:
            for dir in room_in:
                if dir == '左':
                    cv2.arrowedLine(room, (60, 250), (160, 250), (0, 0, 255), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '右':
                    cv2.arrowedLine(room, (440, 250), (340, 250), (0, 0, 255), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '上':
                    cv2.arrowedLine(room, (250, 60), (250, 160), (0, 0, 255), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
                elif dir == '下':
                    cv2.arrowedLine(room, (250, 440), (250, 340), (0, 0, 255), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        # 添加人物进入的方向
        person_direction = self.person_info[person_name]['direction']
        if person_direction == '左':
            cv2.arrowedLine(room, (440, 250), (340, 250), (255, 0, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        elif person_direction == '右':
            cv2.arrowedLine(room, (60, 250), (160, 250), (255, 0, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        elif person_direction == '上':
            cv2.arrowedLine(room, (250, 440), (250, 340), (255, 0, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        elif person_direction == '下':
            cv2.arrowedLine(room, (250, 60), (250, 160), (255, 0, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.3)
        # 添加房间内的人物信息
        color_list = []
        for name, info in self.person_info.items():
            person_lc = info['lc']
            if person_lc == lc:
                color_list.append(info['color'])

        # 按照不同的人数在同一房间中进行人群绘制
        person_cnt = 1
        for color in color_list:
            if person_cnt == 1:
                cv2.circle(room, (250, 250), 10, color, 20)
            elif person_cnt == 2:
                cv2.circle(room, (200, 250), 10, color, 20)
            elif person_cnt == 3:
                cv2.circle(room, (300, 250), 10, color, 20)
            elif person_cnt == 4:
                cv2.circle(room, (250, 200), 10, color, 20)
            elif person_cnt == 5:
                cv2.circle(room, (250, 300), 10, color, 20)
            elif person_cnt == 6:
                cv2.circle(room, (200, 200), 10, color, 20)
            elif person_cnt == 7:
                cv2.circle(room, (300, 300), 10, color, 20)

            person_cnt += 1

        return room

    def run(self, delay=200):
        """
        运行主程序
        """
        cnt = 0
        dm_say_cnt = 0
        arrive_set = set()
        # 当前控制人物
        person_name = '兰'

        while 1:
            # 根据选择的人物更新房间信息
            self.room = self.make_room(person_name)
            cv2.imshow('hctc', self.map)
            cv2.imshow('room', self.room)
            cv2.imshow('panel', self.panel)

            key = cv2.waitKey(0)
            # 退出
            if key == 27:
                print(f'退出模拟，dm操作次数当前为{dm_say_cnt}次')
                logging.info(f'退出模拟，dm操作次数当前为{dm_say_cnt}次')
                exit()
            # 重置
            elif key == ord('右'):
                self.init_map()
                arrive_set = set()
                cnt = 0
                dm_say_cnt = 0
            # 打开或关闭帮助
            elif key == 9:
                self.help = 1 - self.help
                if self.help:
                    cv2.imshow('help', self.help_img)
                else:
                    cv2.destroyWindow('help')
            # 各单位随机行动一步（仅在自动模式下有效）
            elif key == 32 and self.auto:
                cnt += 1
                print(f"==========第{cnt}轮==========")
                logging.info(f"==========第{cnt}轮==========")
                self.act_dict = self.gen_act_dict()
                # 开始一轮的行动
                for order, name_list in self.act_dict:
                    # 如果所有人都到达梓宫
                    if len(arrive_set) == len(self.person_info):
                        print(f'全部成员已到达，dm操作次数为{dm_say_cnt}次')
                        logging.info(f'全部成员已到达，dm操作次数为{dm_say_cnt}次')
                        break
                    # 如果组内任意一个人到达梓宫，说明所有人都已经到达，故跳过整组
                    if name_list[0] in arrive_set:
                        continue
                    # 如果是跟随模式，则每组操作一次,并选择组行动方向
                    if self.follow:
                        dm_say_cnt += 1
                        group_lc = self.person_info[name_list[0]]['lc']
                        new_group_lc, group_direction = self.random_move(group_lc)

                    print(f"第{order}组：")
                    logging.info(f"第{order}组：")
                    for name in name_list:
                        # 更新时间
                        self.hourglass(name)
                        # 到达梓宫的人不进行行动
                        if name in arrive_set:
                            continue
                        else:
                            # 如果不是跟随模式，则每人操作一次
                            if not self.follow:
                                dm_say_cnt += 1
                                lc = self.person_info[name]['lc']
                                new_lc, direction = self.random_move(lc)
                            else:
                                lc = group_lc
                                new_lc = new_group_lc
                                direction = group_direction
                            self.person_info[name]['lc'] = new_lc
                            self.person_info[name]['direction'] = direction
                            if new_lc == [3.5, 3.5]:
                                arrive_set.add(name)
                                print (f"{name}: 从{self.map_dict[str(tuple(lc))]['name']}向{direction}移动到达梓宫")
                                logging.info(f"{name}: 从{self.map_dict[str(tuple(lc))]['name']}向{direction}移动到达梓宫")
                            else:
                                print (f"{name}: 从{self.map_dict[str(tuple(lc))]['name']}向{direction}移动至{self.map_dict[str(tuple(new_lc))]['name']}")
                                logging.info(f"{name}: 从{self.map_dict[str(tuple(lc))]['name']}向{direction}移动至{self.map_dict[str(tuple(new_lc))]['name']}")
                            self.update_person_info()
                            cv2.imshow('hctc', self.map)
                            sub_key = cv2.waitKey(delay)
                            if sub_key == 27:
                                exit()
                # 如果是跟随模式，则需要判断是否相遇，并更新行动优先级
                if self.follow:
                    self.judge_meet()
                else:
                    pass
                # 运行完直接返回循环，不再进行下面的程序
                continue
            # 开启或关闭跟随模式
            elif key == ord('f'):
                self.follow = 1 - self.follow
                if self.follow == True:
                    print ("开启跟随模式")
                    logging.info("开启跟随模式")
                else:
                    print ("关闭跟随模式")
                    logging.info("关闭跟随模式")
            # 查看某人所在的房间
            elif key in range(48, 56):
                person_name = self.person_id_dict[key]
            # 开启关闭自动模式，开局默认不自动
            elif key == ord('r'):
                self.auto = 1 - self.auto
                if self.auto:
                    print ('开启自动行进模式')
                    logging.info('开启自动行进模式')
                else:
                    print ('关闭自动行进模式')
                    logging.info('关闭自动行进模式')
            # 当前人物向上移动
            elif key == ord('w') and not self.auto:
                direction = "上"
                old_time = self.person_info[person_name]['time']
                lc = self.person_info[person_name]['lc']
                out_list = self.map_dict[str(tuple(lc))]['out']
                if direction in out_list:
                    self.hourglass(person_name)
                    old_direction = self.person_info[person_name]['direction']
                    self.act_stack.append((person_name, lc, old_direction, old_time))
                    new_lc, direction = self.move(lc, direction)
                    self.person_info[person_name]['lc'] = new_lc
                    self.person_info[person_name]['direction'] = direction
                    self.judge_meet()
                    dm_say_cnt += 1
                    lc = self.map_dict[str(tuple(lc))]['name']
                    new_lc = self.map_dict[str(tuple(new_lc))]['name']
                    print(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                    logging.info(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                else:
                    print (f'{person_name}: 当前房间无法向{direction}行进')
                    logging.info(f'{person_name}: 当前房间无法向{direction}行进')
                    continue
            # 当前人物向下移动
            elif key == ord('s') and not self.auto:
                direction = "下"
                old_time = self.person_info[person_name]['time']
                lc = self.person_info[person_name]['lc']
                out_list = self.map_dict[str(tuple(lc))]['out']
                if direction in out_list:
                    self.hourglass(person_name)
                    old_direction = self.person_info[person_name]['direction']
                    self.act_stack.append((person_name, lc, old_direction, old_time))
                    new_lc, direction = self.move(lc, direction)
                    self.person_info[person_name]['lc'] = new_lc
                    self.person_info[person_name]['direction'] = direction
                    self.judge_meet()
                    dm_say_cnt += 1
                    lc = self.map_dict[str(tuple(lc))]['name']
                    new_lc = self.map_dict[str(tuple(new_lc))]['name']
                    print(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                    logging.info(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                else:
                    print (f'{person_name}: 当前房间无法向{direction}行进')
                    logging.info(f'{person_name}: 当前房间无法向{direction}行进')
                    continue
            # 当前人物向左移动
            elif key == ord('a') and not self.auto:
                direction = "左"
                old_time = self.person_info[person_name]['time']
                lc = self.person_info[person_name]['lc']
                out_list = self.map_dict[str(tuple(lc))]['out']
                if direction in out_list:
                    self.hourglass(person_name)
                    old_direction = self.person_info[person_name]['direction']
                    self.act_stack.append((person_name, lc, old_direction, old_time))
                    new_lc, direction = self.move(lc, direction)
                    self.person_info[person_name]['lc'] = new_lc
                    self.person_info[person_name]['direction'] = direction
                    self.judge_meet()
                    dm_say_cnt += 1
                    lc = self.map_dict[str(tuple(lc))]['name']
                    new_lc = self.map_dict[str(tuple(new_lc))]['name']
                    print(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                    logging.info(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                else:
                    print (f'{person_name}: 当前房间无法向{direction}行进')
                    logging.info(f'{person_name}: 当前房间无法向{direction}行进')
                    continue
            # 当前人物向右移动
            elif key == ord('d') and not self.auto:
                direction = "右"
                old_time = self.person_info[person_name]['time']
                lc = self.person_info[person_name]['lc']
                out_list = self.map_dict[str(tuple(lc))]['out']
                if direction in out_list:
                    self.hourglass(person_name)
                    old_direction = self.person_info[person_name]['direction']
                    self.act_stack.append((person_name, lc, old_direction, old_time))
                    new_lc, direction = self.move(lc, direction)
                    self.person_info[person_name]['lc'] = new_lc
                    self.person_info[person_name]['direction'] = direction
                    self.judge_meet()
                    dm_say_cnt += 1
                    lc = self.map_dict[str(tuple(lc))]['name']
                    new_lc = self.map_dict[str(tuple(new_lc))]['name']
                    print(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                    logging.info(f"{person_name}: 从{lc}向{direction}移动至{new_lc}")
                else:
                    print (f'{person_name}: 当前房间无法向{direction}行进')
                    logging.info(f'{person_name}: 当前房间无法向{direction}行进')
                    continue
            # 毒性buff
            elif key == ord('p'):
                old_state = self.person_info[person_name]['poision']
                new_state = 1 - old_state
                self.person_info[person_name]['poision'] = new_state
                if new_state == 1:
                    print (f'为{person_name}施加毒药')
                    logging.info(f'为{person_name}施加毒药')
                else:
                    print (f'为{person_name}解毒')
                    logging.info(f'为{person_name}解毒')
                self.update_person_state()
            # 流血buff
            elif key == ord('b'):
                old_state = self.person_info[person_name]['blood']
                new_state = 1 - old_state
                self.person_info[person_name]['blood'] = new_state
                if new_state == 1:
                    print (f'为{person_name}施加流血')
                    logging.info(f'为{person_name}施加流血')
                else:
                    print (f'为{person_name}止血')
                    logging.info(f'为{person_name}止血')
                self.update_person_state()
            # 当前人物精气调整
            elif key == ord('='):
                self.person_info[person_name]['healthy'] += 1
                self.update_person_state()
            elif key == ord('-'):
                self.person_info[person_name]['healthy'] -= 1
                self.update_person_state()
            # 当前人物气脉调整
            elif key == ord('.'):
                self.person_info[person_name]['chi'] += 1
                self.update_person_state()
            elif key == ord(','):
                self.person_info[person_name]['chi'] -= 1
                self.update_person_state()
            # 当前人物神识调整
            elif key == ord('+'):
                self.person_info[person_name]['energy'] += 1
                self.update_person_state()
            elif key == ord('_'):
                self.person_info[person_name]['energy'] -= 1
                self.update_person_state()
            # 自定义选择当前人物的位置
            elif key == ord('u'):
                location = input('请输入当前人物的新位置：')
                if location not in self.room_name_dict.keys():
                    print (f'不存在为{location}的房间，故人物没有成功跳转')
                    logging.info(f'不存在为{location}的房间，故人物没有成功跳转')
                else:
                    lc = self.room_name_dict[location]
                    self.person_info[person_name]['lc'] = lc
            # 清空当前房间的食物
            elif key == ord('o'):
                lc = str(tuple(self.person_info[person_name]['lc']))
                self.map_dict[lc]['food'] = 0
            # 计算命卦
            elif key == ord('m'):
                while 1:
                    year = input('请输入出生年份(输入q退出)：')
                    if year == 'q':
                        break
                    try:
                        year = int(year)
                    except:
                        print ('输入格式有误')
                        continue
                    if 0 < year <= 2900:
                        male_num = (2900 - year) % 9
                        female_num = (year - 1904) % 9

                        if male_num == 5:
                            male_ming = self.minggua[male_num][0]
                        else:
                            male_ming = self.minggua[male_num]

                        if female_num == 5:
                            female_ming = self.minggua[female_num][1]
                        else:
                            female_ming = self.minggua[female_num]

                        for name, info in self.person_info.items():
                            if info['gua'] == male_ming:
                                person1 = name
                            if info['gua'] == female_ming:
                                person2 = name

                        print (f'男： {male_ming}命, 对应角色<{person1}>')
                        print (f'女： {female_ming}命, 对应角色<{person2}>')
                    else:
                        print('输入年份有误')
                        continue

            # 加载人物信息
            elif key == ord('l'):
                self.load_person_state()

            # 隐藏/显示地图
            elif key == ord('h'):
                self.hide = 1 - self.hide

            # 出宫/入宫模式
            elif key == ord('c'):
                self.out = 1 - self.out
                if self.out:
                    # 随机分布鬼子
                    self.random_guizi()
                    print ('当前为出宫模式')
                else:
                    print ('当前为入宫模式')

            # 撤销操作,自动模式下不可撤销
            elif key == ord('z') and not self.auto:
                if dm_say_cnt == 0 or self.act_stack == []:
                    pass
                else:
                    dm_say_cnt -= 1
                    z_name, z_lc, z_direction, z_time = self.act_stack.pop()
                    self.person_info[z_name]['lc'] = z_lc
                    self.person_info[z_name]['direction'] = z_direction
                    self.person_info[z_name]['time'] = z_time
                    person_name = z_name
                    print (f'{z_name}撤销至上一个房间{self.map_dict[str(tuple(z_lc))]["name"]}')

            # 非自动模式下进行一次更新和状态保存,若不隐藏界面则进行地图显示更新
            if self.hide:
                self.map = np.zeros((1200, 1200, 3), np.uint8)
            else:
                self.update_person_info()
            # 保存当前人物状态
            self.save_person_state()
            # cv2.imshow('hctc', self.map)


if __name__ == "__main__":

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=f'data/log.txt', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    hctc = HuangChangTiCou()
    hctc.run(delay=200)
    # hctc.show_map()
