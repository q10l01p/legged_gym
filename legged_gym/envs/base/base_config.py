import inspect

class BaseConfig:
    """
    基础配置类，用于递归地初始化所有成员类

    注解:
    - 该类包含一个构造函数和一个静态方法，用于初始化所有成员类。
    - 忽略所有以'__'开头的名称（内置方法）。
    """

    def __init__(self) -> None:
        """
        类的构造函数，用于初始化所有成员类

        注解:
        - 调用init_member_classes方法来递归初始化成员类。
        """
        self.init_member_classes(self)  # 初始化成员类的实例
    
    @staticmethod
    def init_member_classes(obj):
        """
        静态方法，用于初始化对象的成员类

        参数:
        - obj: 需要初始化成员类的对象

        注解:
        - 遍历对象的所有属性名，忽略以'__class__'为名的内置属性。
        - 对于每个属性，检查是否为类，如果是，则实例化该类，并递归初始化其成员。
        """
        for key in dir(obj):                            # 遍历对象的所有属性名
            if key == "__class__":                      # 忽略'__class__'内置属性
                continue
            var = getattr(obj, key)                     # 获取对应的属性对象
            if inspect.isclass(var):                    # 检查属性是否为类
                i_var = var()                           # 实例化该类
                setattr(obj, key, i_var)                # 将属性设置为该实例
                BaseConfig.init_member_classes(i_var)   # 递归初始化该实例的成员类
