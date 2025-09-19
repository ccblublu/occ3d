NUSC_SEG_MAP = {
    0: "noise",
    1: "animal",
    2: "human.pedestrian.adult",
    3: "human.pedestrian.child",
    4: "human.pedestrian.construction_worker",
    5: "human.pedestrian.personal_mobility",
    6: "human.pedestrian.police_officer",
    7: "human.pedestrian.stroller",
    8: "human.pedestrian.wheelchair",
    9: "movable_object.barrier",
    10: "movable_object.debris",
    11: "movable_object.pushable_pullable",
    12: "movable_object.trafficcone",
    13: "static_object.bicycle_rack",
    14: "vehicle.bicycle",
    15: "vehicle.bus.bendy",
    16: "vehicle.bus.rigid",
    17: "vehicle.car",
    18: "vehicle.construction",
    19: "vehicle.emergency.ambulance",
    20: "vehicle.emergency.police",
    21: "vehicle.motorcycle",
    22: "vehicle.trailer",
    23: "vehicle.truck",
    24: "flat.driveable_surface",
    25: "flat.other",
    26: "flat.sidewalk",
    27: "flat.terrain",
    28: "static.manmade",
    29: "static.other",
    30: "static.vegetation",
    31: "vehicle.ego",
}
NUSC_FINE2COARSE = {
    "noise": "ignore",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
    "flat.driveable_surface": "driveable_surface",
    "flat.sidewalk": "sidewalk",
    "flat.terrain": "terrain",
    "flat.other": "other_flat",
    "static.manmade": "manmade",
    "static.vegetation": "vegetation",
    "static.other": "ignore",
    "vehicle.ego": "ignore",
}

NUSC_COARSE2IDX = {
    "ignore": 0,
    "barrier": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "construction_vehicle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "trailer": 9,
    "truck": 10,
    "driveable_surface": 11,
    "other_flat": 12,
    "sidewalk": 13,
    "terrain": 14,
    "manmade": 15,
    "vegetation": 16,
}
ONTIME2IDX = {
            "bicycle": 5,
            "building": 8,
            "bus": 2,
            "car": 0,
            "cone": 9,
            "crowd": 10,
            "curbside": 11,
            "fence": 12,
            "motorcycle": 4,
            "other_ground": 13,
            "other_object": 14,
            "other_structure": 15,
            "pedestrian": 7,
            "pole": 16,
            "road": 17,
            "tree": 18,
            "tricycle": 6,
            "truck": 1,
            "vegetation": 19,
        }

NUSC_FINEIDX2COARSEIDX = {
    0: 0,
    1: 0,
    2: 7,
    3: 7,
    4: 7,
    5: 0,
    6: 7,
    7: 0,
    8: 0,
    9: 1,
    10: 0,
    11: 0,
    12: 8,
    13: 0,
    14: 2,
    15: 3,
    16: 3,
    17: 4,
    18: 5,
    19: 0,
    20: 0,
    21: 6,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: 0,
    30: 16,
    31: 0,
}

ADE20K_CLASSES = {
    "wall": "manmade",             #? 墙 
    "building": "manmade",         #? 建筑
    "sky": "ignore",              #? 天空
    "floor": "ignore",            #? 地板
    "tree": "vegetation",             #? 树
    "ceiling": "ignore",          #? 天花板
    "road": "driveable_surface",             #? 道路
    "bed ": "ignore",             #? 床
    "windowpane": "ignore",       #? 窗户
    "grass": "ignore",            #? 草
    "cabinet": "ignore",          #? 橱柜
    "sidewalk": "sidewalk",         #? 人行道
    "person": "pedestrian",           #? 人
    "earth": "ignore",            #? 地
    "door": "ignore",             #? 门
    "table": "ignore",            #? 桌子
    "mountain": "ignore",         #? 山
    "plant": "vegetation",            #? 植物
    "curtain": "ignore",          #? 帘子
    "chair": "barrier",            #? 椅子
    "car": "car",              #? 车
    "water": "ignore",            #? 水
    "painting": "ignore",         #? 油画
    "sofa": "ignore",             #? 沙发
    "shelf": "ignore",            #? 书架
    "house": "manmade",            #? 房子
    "sea": "ignore",              #? 海
    "mirror": "ignore",           #? 镜子
    "rug": "ignore",              #? 地毯
    "field": "other_flat",            #? 田地
    "armchair": "barrier",         #? 长椅
    "seat": "ignore",             #? 座位 
    "fence": "ignore",            #? 栅栏
    "desk": "ignore",             #? 书桌
    "rock": "ignore",             #? 岩石
    "wardrobe": "ignore",         #? 衣柜
    "lamp": "ignore",             #? 灯
    "bathtub": "ignore",          #? 浴盆
    "railing": "barrier",          #? 栏杆
    "cushion": "ignore",          #? 垫子
    "base": "ignore",             #? 底座
    "box": "ignore",              #? 盒子
    "column": "barrier",           #? 柱子
    "signboard": "barrier",        #? 标牌
    "chest of drawers": "ignore", #? 柜子
    "counter": "ignore",          #? 柜台
    "sand": "ignore",             #? 沙
    "sink": "ignore",             #? 洗涤槽
    "skyscraper": "manmade",       #? 摩天大楼
    "fireplace": "ignore",        #? 壁炉
    "refrigerator": "ignore",     #? 冰箱
    "grandstand": "ignore",       #? 赛场
    "path": "other_flat",             #? 小径
    "stairs": "ignore",           #? 楼梯
    "runway": "other_flat",           #? 跑道
    "case": "ignore",             #? 案子
    "pool table": "ignore",       #? 桌球桌
    "pillow": "ignore",           #? 枕头
    "screen door": "ignore",      #? 门的屏幕
    "stairway": "ignore",         #? 楼梯
    "river": "ignore",            #? 河
    "bridge": "ignore",           #? 桥
    "bookcase": "ignore",         #? 书架
    "blind": "ignore",            #? 百叶窗
    "coffee table": "ignore",     #? 咖啡桌
    "toilet": "manmade",           #? 厕所
    "flower": "vegetation",           #? 花
    "book": "ignore",             #? 书
    "hill": "ignore",             #? 小山
    "bench": "barrier",            #? 长椅
    "countertop": "ignore",       #? 柜台
    "stove": "ignore",            #? 火炉
    "palm": "vegetation",             #? 棕榈树
    "kitchen island": "ignore",   #? 厨房岛
    "computer": "ignore",         #? 电脑
    "swivel chair": "ignore",     #? 旋转椅
    "boat": "ignore",             #? 船
    "bar": "ignore",              #? 吧台
    "arcade machine": "ignore",   #? 街机
    "hovel": "manmade",            #? 荒屋
    "bus": "bus",              #? 公交车
    "towel": "ignore",            #? 毛巾
    "light": "ignore",            #? 灯
    "truck": "truck",            #? 卡车
    "tower": "ignore",            #? 塔
    "chandelier": "ignore",       #? 吊灯
    "awning": "ignore",           #? 遮阳篷
    "streetlight": "ignore",      #? 街灯
    "booth": "ignore",            #? 橱柜
    "television receiver": "ignore",#? 电视
    "airplane": "ignore",         #? 飞机
    "dirt track": "driveable_surface",       #? 脏土路
    "apparel": "ignore",          #? 服装
    "pole": "ignore",             #? 柱子
    "land": "other_flat",             #? 土地
    "bannister": "barrier",        #? 栏杆
    "escalator": "ignore",        #? 电梯
    "ottoman": "ignore",          #? 席子
    "bottle": "barrier",           #? 瓶子
    "buffet": "ignore",           #? 餐桌
    "poster": "ignore",           #? 海报
    "stage": "ignore",            #? 舞台
    "van": "truck",              #? 货车
    "ship": "ignore",             #? 船
    "fountain": "ignore",         #? 喷泉
    "conveyer belt": "ignore",    #? 传送带
    "canopy": "ignore",           #? 遮阳篷
    "washer": "ignore",           #? 洗衣机
    "plaything": "ignore",        #? 玩具
    "swimming pool": "ignore",    #? 游泳池
    "stool": "barrier",            #? 椅子
    "barrel": "barrier",           #? 桶
    "basket": "ignore",           #? 篮子
    "waterfall": "ignore",        #? 瀑布
    "tent": "ignore",             #? 帐篷
    "bag": "ignore",              #? 包
    "minibike": "motorcycle",         #? 小型摩托车
    "cradle": "ignore",           #? 摇篮
    "oven": "ignore",             #? 炉子
    "ball": "barrier",             #? 球
    "food": "ignore",             #? 食物 
    "step": "ignore",             #? 阶梯
    "tank": "construction_vehicle",             #? 坦克
    "trade name": "ignore",       #? 商标
    "microwave": "ignore",        #? 微波炉
    "pot": "ignore",              #? 锅
    "animal": "ignore",           #? 动物
    "bicycle": "bicycle",          #? 自行车
    "lake": "ignore",             #? 湖
    "dishwasher": "ignore",       #? 洗碗机
    "screen": "ignore",           #? 屏幕
    "blanket": "ignore",          #? 毯子
    "sculpture": "ignore",        #? 雕塑
    "hood": "ignore",             #? 食谱
    "sconce": "ignore",           #? 灯座
    "vase": "ignore",             #? 花瓶
    "traffic light": "manmade",    #? 交通灯
    "tray": "ignore",             #? 托盘
    "ashcan": "barrier",           #? 垃圾桶
    "fan": "ignore",              #? 风扇
    "pier": "ignore",             #? 岛
    "crt screen": "ignore",       #? CRT屏幕
    "plate": "ignore",            #? 盘子
    "monitor": "ignore",          #? 显示器
    "bulletin board": "ignore",   #? 告示牌
    "shower": "ignore",           #? 淋浴
    "radiator": "ignore",         #? 暖气
    "glass": "ignore",            #? 玻璃
    "clock": "ignore",            #? 时钟
    "flag": "ignore",             #? 旗帜
}

ADE20K2NUSC = {
    i: NUSC_COARSE2IDX[ADE20K_CLASSES[k]] for i, k in enumerate(ADE20K_CLASSES)
}
# print(ADE20K2NUSC)