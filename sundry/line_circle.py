import math


class point:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    # 向量
    def sub(self, _p2):
        return point(_p2.x - self.x, _p2.y - self.y)

    # 距离
    def distance(self, _p2):
        return math.sqrt(math.pow(self.x - _p2.x, 2) + math.pow(self.y - _p2.y, 2))

    # 相对方向（象限）
    def direction(self, _p2):
        if self.y < _p2.y:
            if self.x < _p2.x:
                return 1
            else:
                return 2
        else:
            if self.x > _p2.x:
                return 3
            else:
                return 4


class line:
    def __init__(self, _p1, _p2):
        self.p1 = _p1
        self.p2 = _p2
        self.vec = _p1.sub(_p2)

    def inter(self, _pl):
        # 矢量叉乘
        if type(_pl) == line:
            if mult_x(self.p1.sub(_pl.p1), self.p1.sub(_pl.p2)) * mult_x(self.p2.sub(_pl.p1),
                                                                         self.p2.sub(_pl.p2)) >= 0 and mult_x(
                _pl.p1.sub(self.p1), _pl.p1.sub(self.p2)) * mult_x(_pl.p2.sub(self.p1), _pl.p2.sub(self.p2)) >= 0:
                return True
            else:
                return False
        elif type(_pl) == cirle:
            return
            # if self.p1.distance(_pl.c) < _pl.r and self.p2.distance(_pl.c) < _pl.r:
            #     return False
            # elif self.p1.distance(_pl.c) < _pl.r and self.p2.distance(_pl.c) < _pl.r:
            #     return True
            # else:
            #     if _pl.c.direction(self.p1) == _pl.c.direction(self.p1):
            #         if _pl.c.direction(self.p1)
        else:
            ip = 0
            for o in _pl:
                if self.inter(o):
                    ip += 1
            return ip


class cirle:
    def __init__(self, _p1, _p2, _qua, _c):
        self.p1 = _p1
        self.p2 = _p2
        self.c = _c  # 圆心
        self.qua = _qua  # 象限
        self.r = abs(_p2.x - _p1.x)  # 半径

    def inter(self, _pl):
        if type(_pl) == line:
            return
        # 圆弧与圆弧相交
        elif type(_pl) == cirle:
            if self._inter(_pl) and _pl._inter(self):
                return True
            else:
                return False
        else:
            ip = 0
            for o in _pl:
                if self.inter(o):
                    ip += 1
            return ip

    # 圆弧与圆相交
    # 1）同一象限/相对象限 半径在范围 [圆心到端点距离, 圆心距 - 半径] 则相交
    # 2）其他 半径在范围 [圆心到端点距离, 圆心到另一端点距离] 则相交
    def _inter(self, _circle):
        if self.c.direction(_circle.c) == self.qua:
            _range = (self.p1.distance(_circle.c), self.c.distance(_circle.c) - self.r)
        elif (self.c.direction(_circle.c) + 2) % 4 == self.qua:
            _range = (self.c.distance(_circle.c) - self.r, self.p1.distance(_circle.c))
        else:
            _range = (min(self.p1.distance(_circle.c), self.p2.distance(_circle.c)),
                      max(self.p1.distance(_circle.c), self.p2.distance(_circle.c)))

        if _range[0] <= _circle.r <= _range[1]:
            return True
        else:
            return False


# 组合图像
class polyline:
    def __init__(self, _polylist=None):
        if _polylist:
            self.polylist = _polylist
        else:
            self.polylist = []


# 矢量叉乘
def mult_x(v1, v2):
    return v1.x * v2.y - v1.y * v2.x
