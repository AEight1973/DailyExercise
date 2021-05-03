import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


# 链表类
class node():
    def __init__(self, _p, _i):
        self.point = _p
        self.index = _i

    def next(self, sdlist):
        return sdlist[self.index]


class sdlist():
    def __init__(self, _list_n=[]):
        self.list_n = _list_n
        self.index = 0

    def next(self):
        self.index = self.list_n[self.index].index
        return self.list_n[self.index]

    def append(self, p):
        self.list_n[-1].index = len(self.list_n)
        n = node(p, 0)
        self.list_n.append(n)


# 向量类
class vector():
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    # 向量相减
    def sub(self, another):
        IsVector(another)
        newx = self.x - another.x
        newy = self.y - another.y
        return vector(newx, newy)

    # 向量相加
    def plus(self, another):
        IsVector(another)
        newx = self.x + another.x
        newy = self.y + another.y
        return vector(newx, newy)

    # 向量数乘
    def mult_num(self, m):
        return vector(self.x * m, self.y * m)

    # 向量叉乘
    def mult_x(self, another):
        IsVector(another)
        return self.x * another.y - self.y * another.x

    # 向量点乘
    def mult_d(self, another):
        IsVector(another)
        return self.x * another.x + self.y * another.y

    # 向量的模
    def value(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    # 向量的单位向量
    def unit(self):
        return vector(self.x / self.value(), self.y / self.value())

    # 向量的左垂直向量
    def vert_left(self, m=1):
        return vector(-self.y, self.x).unit().mult_num(m)

    # 向量的右垂直向量
    def vert_right(self, m=1):
        return vector(self.y, -self.x).unit().mult_num(m)

    # 两向量夹角
    def angle(self, another=None):
        if another == None:
            return math.acos(self.x / self.value()) * (self.y / abs(self.y)) * 180 / math.pi
        else:
            IsVector(another)
            return math.acos(self.mult_d(another) / (self.value() * another().value())) * (
                        self.y / abs(self.y)) * 180 / math.pi


# 点类
class point():
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    # 两点生成向量
    def ToVector(self, another=None):
        if another == None:
            return vector(self.x, self.y)
        else:
            newx = self.x - another.x
            newy = self.y - another.y
            return vector(newx, newy)

    # 点根据向量移动
    def plus_vector(self, v):
        IsVector(v)
        return point(self.x + v.x, self.y + v.y)

    # 两点之间距离
    def distance(self, another):
        IsPoint(another)
        return ((self.x - another.x) ** 2 + (self.y - another.y) ** 2) ** 0.5

    # 打印点
    def print(self):
        print("( {:.2f}, {:.2f} )".format(self.x, self.y))

    # 画点
    def draw(self):
        plt.plot(self.x, self.y, 'ro')


# 线类
class line():
    def __init__(self, _sp, _ep):
        self.startpoint = _sp
        self.endpoint = _ep
        self.vector = vector(_ep.x - _sp.x, _ep.y - _sp.y)

    # 两线交点
    def point_inter(self, another):
        IsLine(another)
        if Intersect_Line(self, another):
            area1 = abs(
                self.startpoint.ToVector(another.startpoint).mult_x(self.startpoint.ToVector(another.endpoint))) / 2
            area2 = abs(self.endpoint.ToVector(another.startpoint).mult_x(self.endpoint.ToVector(another.endpoint))) / 2
            newx = (area1 * self.endpoint.x - area2 * self.startpoint.x) / (area1 - area2)
            newy = (self.vector.y / self.vector.x) * (newx - self.startpoint.x) + self.startpoint.y
            return point(newx, newy)

    # 线方向翻转
    def reverse(self):
        newsp = self.endpoint
        newep = self.startpoint
        return line(newep, newsp)

    # 打印线
    def print(self):
        print("line( ( {:.2f}, {:.2f} ), ( {:.2f}, {:.2f} ) )".format(self.startpoint.x, self.startpoint.y,
                                                                      self.endpoint.x, self.endpoint.y))

    # 画线
    def draw(self):
        plt.plot([self.startpoint.x, self.endpoint.x], [self.startpoint.y, self.endpoint.y], "b")


# 折线类
class polyline():
    def __init__(self, _list_p=[]):
        if not isinstance(_list_p, list):
            raise ValueError("not an object of class list")
        elif len(_list_p) < 2:
            raise ValueError("require at least two points in this list")
        else:
            for p in _list_p:
                IsPoint(p)
        self.list_p = _list_p
        self.side = len(_list_p)
        _list_l = []
        for i in range(len(self.list_p) - 1):
            point1 = self.list_p[i]
            point2 = self.list_p[i + 1]
            _list_l.append(line(point1, point2))
        self.list_l = _list_l

    # 折线段闭合为多边形
    def ToPolygon(self):
        if self.side == 2:
            raise ValueError("require at least three points in this list")
        else:
            return polygon(self.list_p)

    def draw(self):
        for line_i in self.list_l:
            line_i.draw()

    # 打印折线段
    def print(self):
        print("polyline(")
        for line_i in self.list_l:
            line_i.print()
        print(")")


# 简单多边形类
class polygon():
    def __init__(self, _list_p=[]):
        if not isinstance(_list_p, list):
            raise ValueError("not an object of class list")
        elif len(_list_p) < 3:
            raise ValueError("require at least three points in this list")
        else:
            for p in _list_p:
                IsPoint(p)
        self.list_p = _list_p
        self.side = len(_list_p)
        _list_l = []
        for i in range(len(self.list_p) - 1):
            point1 = self.list_p[i]
            point2 = self.list_p[i + 1]
            _list_l.append(line(point1, point2))
        _list_l.append(line(point2, self.list_p[0]))
        self.list_l = _list_l
        i = 1
        _list_n = []
        for p in _list_p:
            node_i = node(p, i)
            _list_n.append(node_i)
            i += 1
        _list_n[i - 2].index = 0
        self.list_n = _list_n

    # 多边形的范围
    def content(self):
        content = {"xmin": self.list_p[0].x, "xmax": self.list_p[0].x, "ymin": self.list_p[0].y,
                   "ymax": self.list_p[0].y}
        for p in self.list_p:
            if p.x < content["xmin"]:
                content["xmin"] = p.x
            elif p.x > content["xmax"]:
                content["xmax"] = p.x
            if p.y < content["ymin"]:
                content["ymin"] = p.y
            elif p.y > content["ymax"]:
                content["ymax"] = p.y
        return content

    def print(self):
        print("polygon(")
        for p in self.list_p:
            p.print()
        print(")")


# 圆类
class circle():
    def __init__(self, _cp, _r):
        self.centerpoint = _cp
        self.r = _r

    # 画圆
    def draw(self):
        cir = Circle(xy=(self.centerpoint.x, self.centerpoint.y), radius=self.r)
        ax.add_patch(cir)


# 矩形类
class rectangle():
    def __init__(self, _p, _w, _h, _a):
        self.point = _p
        self.width = _w
        self.height = _h
        self.angle = _a

    # 画矩形
    def draw(self):
        rec = Rectangle((self.point.x, self.point.y), self.width, self.height, angle=self.angle)
        ax.add_patch(rec)


# 复杂多边形类
class polygon_complex():
    def __init__(self, _list_la=[]):
        self.list_la = _list_la

    # 打印复杂多边形
    def print(self):
        for pln in self.list_la:
            pln.print()


# 检测类
def IsVector(v):
    if not isinstance(v, vector):
        raise ValueError("not an object of class vector")


def IsPoint(v):
    if not isinstance(v, point):
        raise ValueError("not an object of class point")


def IsLine(v):
    if not isinstance(v, line):
        raise ValueError("not an object of class line")


# 折线段拐向
def turn_Line(v1, v2):
    IsVector(v1)
    IsVector(v2)
    value = v1.mult_x(v2)
    if value > 0:
        return "left"
    elif value < 0:
        return "right"
    elif v1.unit() == v1.unit():
        return "collinear"
    else:
        return "reverse"


# 点是否在线上
def On_Line(p, l):
    IsPoint(p)
    IsLine(l)
    v0 = p.ToVector()
    v1 = l.startpoint.ToVector()
    v2 = l.endpoint.ToVector()
    value = v2.sub(v0).mult_x(v1.sub(v0))
    if value == 0:
        return True
    else:
        return False


# 两直线段是否相交
def Intersect_Line(l1, l2):
    p1 = l1.startpoint
    p2 = l1.endpoint
    p3 = l2.startpoint
    p4 = l2.endpoint
    value1 = p4.ToVector(p1).mult_x(p3.ToVector(p1))
    value2 = p4.ToVector(p2).mult_x(p3.ToVector(p2))
    value3 = p2.ToVector(p3).mult_x(p1.ToVector(p3))
    value4 = p2.ToVector(p4).mult_x(p1.ToVector(p4))
    if value1 * value2 == 0 and value3 * value4 == 0:
        if On_Line(p1, l2) or On_Line(p2, l2):
            return True
        else:
            return False
    elif value1 * value2 <= 0 and value3 * value4 <= 0:
        return True
    else:
        return False


def Intersect_Polyline(l, pl):
    for line_i in pl.list_l:
        if Intersect_Line(l, line_i):
            return True
    return False


# 点是否在多边形内
def contain_polygan(p, polygon):
    line0 = line(p, point(p.x, polygon.content()["ymin"]))
    n_inter = 0
    for line_i in polygon.list_l:
        if On_Line(p, line_i):
            return False
        elif Intersect_Line(line0, line_i):
            n_inter += 1
    if n_inter // 2 == n_inter:
        return False
    else:
        return True


# 模拟器
class simulation():
    def __init__(self):
        pass

    def point(self):
        x = random.randint(0, 100) / 10
        y = random.randint(0, 100) / 10
        return point(x, y)

    def vector(self):
        point1 = self.point()
        point2 = self.point()
        return point1.ToVector(point2)

    def line(self):
        point1 = self.point()
        point2 = self.point()
        return line(point1, point2)

    def polygon(self):
        line0 = self.line()
        list_p = [line0.startpoint, line0.endpoint]
        point0 = self.point()
        list_p.append(point0)
        point_i = list_p[-1]
        n = random.randint(2, 10)
        for i in range(n - 2):
            point_i = self.point()
            while Intersect_Polyline(line(list_p[-1], point_i), polyline(list_p[:-1])):
                point_i = self.point()
            list_p.append(point_i)
        while Intersect_Polyline(line(list_p[0], point_i), polyline(list_p[1:-1])):
            point_i = self.point()
        list_p[-1] = point_i
        return polygon(list_p)

    def polyline(self, n=None):
        line0 = self.line()
        list_p = [line0.startpoint, line0.endpoint]
        point0 = self.point()
        list_p.append(point0)
        point_i = list_p[-1]
        if n == None:
            n = random.randint(2, 5)
        for i in range(n - 2):
            point_i = self.point()
            while Intersect_Polyline(line(list_p[-1], point_i), polyline(list_p[:-1])):
                point_i = self.point()
            list_p.append(point_i)
        return polyline(list_p)


# 折线段缓冲区
def buffer_polyline(pl, d):
    buffer = []
    for p in pl.list_p:
        circle_i = circle(p, d)
        buffer.append(circle_i)
        circle_i.draw()
    for l in pl.list_l:
        p = l.startpoint.plus_vector(l.vector.vert_right(d))
        w = l.vector.value()
        h = 2 * d
        a = l.vector.angle()
        rectangle_i = rectangle(p, w, h, a)
        buffer.append(rectangle_i)
        rectangle_i.draw()
    return polygon_complex(buffer)


# 主程序（折线段拐向）
polyline1 = simulation().polyline(2)
vector1 = polyline1.list_l[0].vector
vector2 = polyline1.list_l[1].vector
turn = turn_Line(vector1, vector2)
print("此为检测1（折线段拐向）\n随机折线段为：")
polyline1.print()
print("拐向为{}".format(turn))

# 主程序（点是否在线上）
point1 = simulation().point()
line1 = simulation().line()
OnlineorNot = On_Line(point1, line1)
print("此为检测2（点是否在线上）\n随机点为：")
point1.print()
print("随机线为：")
line1.print()
if OnlineorNot:
    print("点在线上")
else:
    print("点不在线上")

# 主程序（两直线段相交及其交点）
line2 = simulation().line()
line3 = simulation().line()
IntersectorNot = Intersect_Line(line2, line3)
print("此为检测3（两直线段相交及其交点）\n随机直线段为：")
line2.print()
line3.print()
if IntersectorNot:
    print("两直线段相交，且交点为：")
    point2 = line2.point_inter(line3)
    point2.print()
else:
    print("两直线段不相交")

# 主程序（点是否在多边形内）
point3 = simulation().point()
polygon1 = simulation().polygon()
InpolygonorNot = contain_polygan(point3, polygon1)
print("此为检测4（点是否在多边形内）\n随机点为：")
point3.print()
print("随机多边形为：")
polygon1.print()
if IntersectorNot:
    print("点在多边形内")
else:
    print("点不在多边形内")

# 主程序（折线段的缓冲区）
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(-2, 12)
plt.ylim(-2, 12)
polyline2 = simulation().polyline()
print("此为检测5（折线段的缓冲区）\n随机折线段为：")
polyline2.print()
polyline2.draw()
d = 1
buffer = buffer_polyline(polyline2, d)
plt.show()
print("缓冲区请见图")
