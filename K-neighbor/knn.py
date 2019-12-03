import numpy as np

class KDT(object):
    def __init__(self):
        self.head = None


class KDTNode(object):
    def __init__(self, parent, value, axis):
        self.left = None
        self.right = None
        self.parent = parent
        self.value = value
        self.axis = axis
        self.isVisit = False  # 搜索时记录是否被访问过

    def isLeaf(self):
        return self.left == None & self.right == None


def handleData(data, idx):
    '''
    将数据根据idx维度的中位数进行划分
    :param data: 要划分的数据
    :param idx:  在数据的哪个维度上进行划分
    :return: 作为划分结点的med ,左子树数据left,右子树数据right
    '''
    tmp = data[:, idx]
    medI = np.median(tmp)
    no = (np.abs(tmp - medI)).argmin()
    medI = data[no, idx]
    med = data[no]
    data = np.delete(data, no, axis=0)
    left = data[data[:, idx] < medI]
    right = data[data[:, idx] >= medI]
    return med, left, right


def builtNode(data, parent, idx):
    '''
    构造kd树的结点
    :param data: 输入的数据
    :param parent: 父节点
    :param idx: 在idx维度上进行划分
    :return:
    '''
    # 将数据分为idx维度上的中间节点，左部分，右部分
    med, left, right = handleData(data, idx)
    node = KDTNode(parent=parent, value=med, axis=idx)
    if (left.size != 0):
        node.left = builtNode(left, node, 1 - idx)
    if (right.size != 0):
        node.right = builtNode(right, node, 1 - idx)
    return node


def builtKDT(data):
    '''
    构造kd树
    :param data: 输入的数据
    :return:
    '''
    kdt = KDT()
    kdt.root = builtNode(data, None, 0)
    return kdt


def printKDT(kdt):
    '''
    二叉树的先序输出
    :param kdt:
    :return:
    '''
    printNode(kdt.root)


def printNode(node):
    print('结点数据:{},划分维度:{}'.format(node.value, node.axis))
    if node.left != None:
        printNode(node.left)
    if node.right != None:
        printNode(node.right)


def findleaf(root, x):
    '''
    根据kd树,找到x对应的叶节点
    :param root: kd树的根节点
    :param x: 目标节点
    :return:  叶节点
    '''
    axis = root.axis
    leaf = root
    if (x[axis] < root.value[axis]) & (root.left != None):
        leaf = findleaf(root.left, x)
    if (x[axis] >= root.value[axis]) & (root.right != None):
        leaf = findleaf(root.right, x)
    return leaf


def nearstNeighborSearch(node, x, curDis, cur, searchSibling):
    '''
    递归的查找最近邻点
    :param node:
    :param x:
    :param curDis:
    :param cur:
    :param searchSibling:
    :return:
    '''
    if node.isVisit:
        return curDis, cur
    node.isVisit = True
    dis = np.sqrt(np.sum((node.value - x) ** 2))
    if dis < curDis:
        curDis = dis
        cur = node
    if (searchSibling == 1) & (node.left != None):
        tmpDis, tmp = nearstNeighbor(node.left, x, curDis=curDis, cur=cur)
        if tmpDis < curDis:
            curDis = tmpDis
            cur = tmp

    if (searchSibling == 2) & (node.right != None):
        tmpDis, tmp = nearstNeighbor(node.right, x, curDis=curDis, cur=cur)
        if tmpDis < curDis:
            curDis = tmpDis
            cur = tmp
    if node.parent != None:
        axis = node.parent.axis
        tmpDis, tmp = 0, None
        if (x[axis] < node.parent.value[axis]) & (node.parent.value[axis] - x[axis] < curDis):
            tmpDis, tmp = nearstNeighborSearch(node.parent, x, curDis, cur, 2)
        elif (x[axis] >= node.parent.value[axis]) & (x[axis] - node.parent.value[axis] < curDis):
            tmpDis, tmp = nearstNeighborSearch(node.parent, x, curDis, cur, 1)
        else:
            tmpDis, tmp = nearstNeighborSearch(node.parent, x, curDis, cur, 0)
        if tmpDis < curDis:
            curDis = tmpDis
            cur = tmp
    return curDis, cur


def nearstNeighbor(root, x, curDis=np.inf, cur=None):
    leaf = findleaf(root, x)
    curDis, cur = nearstNeighborSearch(leaf, x, curDis, cur, 0)
    return curDis, cur


data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
kdt = builtKDT(data)
printKDT(kdt)
x = np.array([4, 5])
nearestDis, nearestNode = nearstNeighbor(kdt.root, x, curDis=np.inf, cur=None)
print(nearestDis, nearestNode.value)
