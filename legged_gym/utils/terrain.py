import numpy as np
from numpy.random import choice

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg as Cfg


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0) -> None:
        # 初始化函数，接受地形配置(cfg)，机器人数量(num_robots)，评估配置(eval_cfg)，评估时的机器人数量(num_eval_robots)

        self.cfg = cfg  # 将地形配置保存到实例变量中
        self.eval_cfg = eval_cfg  # 将评估配置保存到实例变量中
        self.num_robots = num_robots  # 将机器人数量保存到实例变量中
        self.type = cfg.mesh_type  # 从配置中获取地形的网格类型并保存
        if self.type in ["none", 'plane']:  # 如果地形类型为"none"或"plane"，则不进行后续操作
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()  # 加载训练和评估配置中的行和列
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)  # 计算总行数
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))  # 计算最大列数
        self.cfg.env_length = cfg.terrain_length  # 设置环境的长度
        self.cfg.env_width = cfg.terrain_width  # 设置环境的宽度

        self.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)  # 创建一个全为零的高度场数组

        self.initialize_terrains()  # 初始化地形

        self.heightsamples = self.height_field_raw  # 将原始高度场赋值给高度样本
        if self.type == "trimesh":  # 如果地形类型为"trimesh"
            # 将高度场转换为三角网格
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        # 定义加载配置的方法
        self._load_cfg(self.cfg)  # 调用一个内部方法来加载训练配置
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)  # 根据总行数设置行索引
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)  # 根据总列数设置列索引
        self.cfg.x_offset = 0  # 设置水平偏移量为0
        self.cfg.rows_offset = 0  # 设置行偏移量为0
        if self.eval_cfg is None:  # 如果评估配置为空
            return self.cfg.row_indices, self.cfg.col_indices, [], []  # 返回训练配置的行列索引，评估配置的行列索引为空
        else:  # 如果评估配置不为空
            self._load_cfg(self.eval_cfg)  # 调用内部方法加载评估配置
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows,
                                                  self.cfg.tot_rows + self.eval_cfg.tot_rows)  # 设置评估配置的行索引
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)  # 设置评估配置的列索引
            self.eval_cfg.x_offset = self.cfg.tot_rows  # 设置评估配置的水平偏移量
            self.eval_cfg.rows_offset = self.cfg.num_rows  # 设置评估配置的行偏移量
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices  # 返回训练和评估配置的行列索引

    def _load_cfg(self, cfg):
        # 定义一个内部方法，用于加载和设置配置对象的属性

        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]
        # 计算地形类型的累积比例，用于后续确定各类型地形的分布

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 计算子地形的总数，即行数乘以列数

        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # 初始化每个子地形的原点坐标数组，每个原点坐标为三维(x, y, z)，初始为0

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        # 计算每个环境单位在水平方向上的像素数量（根据地形长度和水平比例尺）

        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)
        # 计算每个环境单位在垂直方向上的像素数量（根据地形宽度和水平比例尺）

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        # 计算地形边界的像素宽度（根据边界大小和水平比例尺）

        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        # 计算总列数，包括所有子地形列和两边的边界

        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border
        # 计算总行数，包括所有子地形行和两边的边界

    def initialize_terrains(self):
        # 初始化地形的公共接口方法

        self._initialize_terrain(self.cfg)  # 初始化训练配置的地形
        if self.eval_cfg is not None:  # 如果存在评估配置
            self._initialize_terrain(self.eval_cfg)  # 则也初始化评估配置的地形

    def _initialize_terrain(self, cfg):
        # 定义一个内部方法，根据给定配置初始化地形

        if cfg.curriculum:
            self.curriculum(cfg)  # 如果配置中指定了课程学习方法，则调用curriculum方法来初始化地形
        elif cfg.selected:
            self.selected_terrain(cfg)  # 如果配置中指定了选择特定地形，则调用selected_terrain方法来初始化
        else:
            self.randomized_terrain(cfg)  # 如果上述都不是，则调用randomized_terrain方法随机初始化地形

    def randomized_terrain(self, cfg):
        # 定义一个方法，根据随机逻辑初始化地形

        for k in range(cfg.num_sub_terrains):
            # 遍历所有子地形

            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            # 将一维索引k转换为二维索引(i, j)，对应于地形的行和列

            choice = np.random.uniform(0, 1)
            # 随机生成一个介于0和1之间的数，用于后续选择地形类型

            difficulty = np.random.choice([0.5, 0.75, 0.9])
            # 随机选择一个难度值

            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            # 根据随机选取的参数创建地形

            self.add_terrain_to_map(cfg, terrain, i, j)
            # 将创建的地形添加到地图上的相应位置

    def curriculum(self, cfg):
        # 定义一个方法，根据课程学习的逻辑初始化地形

        for j in range(cfg.num_cols):
            # 遍历所有列

            for i in range(cfg.num_rows):
                # 遍历所有行

                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                # 计算当前子地形的难度，基于它的行索引，实现难度递增

                choice = j / cfg.num_cols + 0.001
                # 计算地形类型的选择值，基于它的列索引，确保每列有所不同

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                # 根据计算出的参数创建地形

                self.add_terrain_to_map(cfg, terrain, i, j)
                # 将创建的地形添加到地图上的相应位置

    def selected_terrain(self, cfg):
        # 定义一个方法，用于初始化用户指定的地形类型

        terrain_type = cfg.terrain_kwargs.pop('type')
        # 从配置中提取并删除地形类型信息

        for k in range(cfg.num_sub_terrains):
            # 遍历所有子地形

            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))
            # 将一维索引k转换为二维索引(i, j)，对应于地形的行和列

            # 创建一个SubTerrain实例，设置其宽度、长度、垂直和水平比例尺
            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            # 使用eval执行地形类型对应的函数，传入地形实例和额外的地形参数

            self.add_terrain_to_map(cfg, terrain, i, j)
            # 将创建的地形添加到地图上的相应位置

    # 定义一个名为 make_terrain 的方法，它接收配置(cfg)、选择(choice)、难度(difficulty)和比例(proportions)四个参数。
    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain("terrain",  # 创建一个 SubTerrain 对象，命名为 "terrain"。
                                           width=cfg.width_per_env_pixels,  # 设置地形的宽度为配置中的每个环境像素的宽度。
                                           length=cfg.width_per_env_pixels,  # 设置地形的长度，与宽度相同。
                                           vertical_scale=cfg.vertical_scale,  # 设置地形的垂直缩放比例。
                                           horizontal_scale=cfg.horizontal_scale)  # 设置地形的水平缩放比例。
        slope = difficulty * 0.4  # 根据难度设置坡度。
        step_height = 0.05 + 0.18 * difficulty  # 根据难度计算台阶的高度。
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)  # 根据难度计算离散障碍物的高度。
        stepping_stones_size = 1.5 * (1.05 - difficulty)  # 根据难度计算踏石的大小。
        stone_distance = 0.05 if difficulty == 0 else 0.1  # 根据难度设置踏石间的距离。
        if choice < proportions[0]:  # 根据选择和比例决定地形类型。
            if choice < proportions[0] / 2:  # 如果选择值较小，则设置坡度为负值。
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)  # 创建有坡度的金字塔形地形。
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)  # 再次创建有坡度的金字塔形地形
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,  # 创建随机平坦的地形。
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice < proportions[2]:
                step_height *= -1  # 如果选择在特定范围内，将台阶高度设为负值。
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height,
                                                 platform_size=3.)  # 创建有台阶的金字塔形地形。
        elif choice < proportions[4]:
            num_rectangles = 20  # 设置矩形障碍物的数量。
            rectangle_min_size = 1.  # 设置矩形障碍物的最小尺寸。
            rectangle_max_size = 2.  # 设置矩形障碍物的最大尺寸。
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     # 创建具有离散障碍物的地形。
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,  # 创建踏石地形。
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            pass  # 对于这个选择范围，不创建任何地形。
        elif choice < proportions[7]:
            pass  # 同上，不进行操作。
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,  # 创建具有随机噪声的地形。
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,  # 再次创建随机平坦地形。
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0  # 修改地形的一半高度字段为0。

        return terrain  # 返回创建的地形对象。

    # 定义一个名为 add_terrain_to_map 的方法，接受配置(cfg)、地形(terrain)、行(row)和列(col)作为参数。
    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row  # 将传入的行参数赋值给变量 i。
        j = col  # 将传入的列参数赋值给变量 j。
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset  # 计算地形在地图上的起始 x 坐标。
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset  # 计算地形在地图上的结束 x 坐标。
        start_y = cfg.border + j * cfg.width_per_env_pixels  # 计算地形在地图上的起始 y 坐标。
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels  # 计算地形在地图上的结束 y 坐标。
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw  # 将地形的高度字段添加到地图的对应位置。

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale  # 计算环境原点的 x 坐标。
        env_origin_y = (j + 0.5) * cfg.terrain_width  # 计算环境原点的 y 坐标。
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset  # 计算地形边界的一个 x 坐标。
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset  # 计算地形边界的另一个 x 坐标。
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)  # 计算地形边界的一个 y 坐标。
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)  # 计算地形边界的另一个 y 坐标。
        # 计算环境原点的 z 坐标，它是当前地块的最大高度乘以垂直缩放比例。
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]  # 在配置的环境原点矩阵中设置当前环境原点的坐标。

    def gap_terrain(terrain, gap_size, platform_size=1.):
        """在给定的地形上创建一个含有间隙的平台"""
        # 根据地形的水平比例调整间隙大小
        gap_size = int(gap_size / terrain.horizontal_scale)
        # 根据地形的水平比例调整平台大小
        platform_size = int(platform_size / terrain.horizontal_scale)

        # 计算地形中心点的坐标
        center_x = terrain.length // 2
        center_y = terrain.width // 2
        # 计算间隙开始前平台的长度
        x1 = (terrain.length - platform_size) // 2
        # 间隙的终点坐标
        x2 = x1 + gap_size
        # 计算间隙开始前平台的宽度
        y1 = (terrain.width - platform_size) // 2
        # 间隙的终点坐标
        y2 = y1 + gap_size

        # 在地形的高度场中创建深坑，深度为-1000，作为间隙的标记
        terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
        # 在间隙周围创建平台，将高度设置为0
        terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0

    def pit_terrain(terrain, depth, platform_size=1.):
        """基于中心点创建一个指定深度和平台大小的坑"""
        # 将深度值按地形的垂直比例调整，确保深度与地形比例相符
        depth = int(depth / terrain.vertical_scale)

        # 计算平台大小，考虑地形的水平比例，并将平台大小调整为距离中心的半径值
        platform_size = int(platform_size / terrain.horizontal_scale / 2)

        # 计算平台在地形长度上的起止点，以地形中心为基点，向两侧延伸
        x1 = terrain.length // 2 - platform_size
        x2 = terrain.length // 2 + platform_size

        # 计算平台在地形宽度上的起止点，以地形中心为基点，向两侧延伸
        y1 = terrain.width // 2 - platform_size
        y2 = terrain.width // 2 + platform_size

        # 在地形的高度场数组中，将指定区域的高度值设置为负深度值，以创建坑
        terrain.height_field_raw[x1:x2, y1:y2] = -depth

    def get_height(self):
        return self.heightsamples
