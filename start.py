from datetime import datetime
import json
import pickle
from mpi4py import MPI
import numpy as np
from functools import lru_cache, reduce
import pandas as pd
import tushare as ts
import threading
import datetime
import math
import multiprocessing as mp
import winsound
from multiprocessing import Pool
from multiprocessing import Process, Queue
import os, time, random
import subprocess
import time, threading
import sys
from queue import Queue
from functools import reduce
import gc

class Sample:
    max_sample = 100 # 类属性
    
    def __init__(self, name='weight sample'):
        """
        类的初始化，一般用来初始化类成员, 不一定需要重写。
        self 关键字是指被创建后的类自己，每一个 self 代表一个被创建的类。
        """
        self.name = name # 创建类成员
        self.samples = [] # 创建类成员
        return
        
    def __repr__(self):
        """
        重写类的表达式，不一定需要重写。
        """
        return "My name is:" + str(self.name) + ", and I have " + str(len(self.samples)) + " samples"
    
    def __getitem__(self, index):
        """
        重写并且添加原本不支持的类索引，不一定需要重写。
        """
        try:
            return self.samples[index]
        except IndexError as error:
            print("No sample with index: " + str(index) + ", err: " + str(error))
            return None
    
    @property
    def sample_size(self):
        """
        类的方法变成员，通过 @property
        变成员的类方法不能传除了 self 的参数
        """
        return len(self.samples)
    
    @classmethod
    def calculate_average(cls, samples):
        """
        静态类方法，不需要创建类就可以调用
        这里注意不是 self 参数，而是 cls 参数，cls 参数直接通过类名字 Sample 就可以调用
        比如: Sample.calculate_average()
        """
        return np.mean(samples)
    
    def collect_samples(self, samples):
        """
        动态类方法
        """
        for sample in samples:
            if self.sample_size >= self.max_sample:
                print("Sample size larger than: " + str(self.max_sample))
                break
            self.samples.append(sample)
        return
            
    def _analyze(self):
        """
        隐藏动态类方法
        不应该从外部调用的类方法，下划线开头的名称表示专门用于类的内部调用
        但是 Python 不阻止你从外部调用
        """
        
        average_weight = self.calculate_average(self.samples)
        std_weight = np.std(self.samples)

        return self.sample_size, average_weight, std_weight

    def get_analytics_report(self):
        """
        动态类方法
        """
        results = self._analyze()
        print("Number of samples: " + str(results[0]))
        print("Average weight: " + "{0:.2f}".format(results[1]))
        print("Standard deviation of weight: " + "{0:.2f}".format(results[2]))
        return

    def __del__(self):
        """
        释放类, 不一定要重写
        """
        print(self.name + " with sample size " + str(self.sample_size) + " is being released")

class Student(object):

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self,value):
        if not isinstance(value, int):
            raise ValueError('分数必须是整数才行呐')
        if value < 0 or value > 100:
            raise ValueError('分数必须0-100之间')
        self._score = value

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def buildTree(preorder, inorder):
    if not preorder:
        return None
    r = preorder[0]
    root = TreeNode(r)
    mid = inorder.index(r)
    root.left = buildTree(preorder[1:1 + mid], inorder[:mid])
    root.right = buildTree(preorder[1 + mid:], inorder[1 + mid:])
    return root

def minArray(numbers):
    n = len(numbers)
    if n == 1: return numbers[0]
    if n == 2: return numbers[0] if numbers[0] < numbers[1] else numbers[1]
    if numbers[-1] > numbers[0]: return numbers[0]
    mid = int(n // 2)
    if numbers[mid] > numbers[-1]:
        return minArray(numbers[mid + 1:])
    elif numbers[mid] < numbers[-1]:
        return minArray(numbers[:mid + 1])
    else:
        return minArray(numbers[:-1])

def findContinuousSequence(target):
    if target == 1:
        return [1]
    if target == 2:
        return []
    ans = []
    for i in range(target // 2 + 1, 1, -1):
        n = target / i - (1 + i) / 2
        if n % 1 == 0 and n >= 0:
            start, end = int(n) + 1, int(n) + i
            ans.append(list(range(start, end + 1)))
    return ans

def raise_test():
    raise ZeroDivisionError('wnmnb!')

def yield_test():
    i = 0
    while i < 5:
        yield i
        i +=1

@lru_cache()
def lru_cache_test(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return lru_cache_test(n - 1) + lru_cache_test(n - 2)

#start = time.time()
#print('%.2f sec'%(time.time() - start))

#i = 1
#for i in range(9):
#    if i == 3: break
'''输出居然是3！！'''
#print(i)

def canThreePartsEqualSum(A):
    s = sum(A)
    n = len(A)
    if s % 3 != 0:
        return False
    t = s // 3
    s1 = 0
    s2 = 0
    for i in range(n - 2):
        s1 += A[i]
        if s1 != t:
            continue
        for j in range(n - 1, i + 1, -1):
            s2 += A[j]
            if s2 == t:
                return True
    return False

#duration = 3000
#freq = 660  # Hz
#winsound.Beep(freq, duration)

from sklearn.model_selection import validation_curve

def gcdOfStrings(str1, str2):
    for i in range(min(len(str1), len(str2)), 0, -1):
        if len(str1) % i == 0 and len(str2) % i == 0:
            if str1[:i] * (len(str1) // i) == str1 and str1[:i] * (len(str2) // i) == str2:
                return str1[:i]
    return ''

result = []
def nQueenBacktrack(path, choice, n):
    #global result
    if len(path) == n:
        result.append(path.copy())
        return

    for i in range(n):
        row = len(choice)
        on = False
        for e in choice:
            if (i == e[1]) or\
                (abs(abs((i - e[1]) / (row - e[0])) - 1) < 1e-9):
                on = True
                break
        if on: continue
        step = ['.'] * n
        step[i] = 'Q'
        step = ''.join(step)
        path.append(step)
        choice.append((row, i))
        nQueenBacktrack(path, choice, n)
        path.pop()
        choice.pop()

def nQueen(n):
    #global result
    path, choice = [], []
    nQueenBacktrack(path, choice, n)
    return result

def permuteUniqueBacktrack(path, idx_path, nums, n):
    if len(path) == n:
        result.append(path.copy())
        return
    
    indicator = []
    for i in range(n):
        if nums[i] in indicator:#判断是否选择过此数
            continue
        if (nums[i] in path) and (i in idx_path):#判断当前元素是否可以加入路径
            continue
        path.append(nums[i])
        indicator.append(nums[i])
        idx_path.append(i)
        permuteUniqueBacktrack(path, idx_path, nums, n)
        path.pop()
        idx_path.pop()

def permuteUnique(nums):
    path, idx_path = [], []
    n = len(nums)
    permuteUniqueBacktrack(path, idx_path, nums, n)
    return result

tel = {'2':'abc', '3':'def', '4':'ghi', '5':'jkl',
        '6':'mno', '7':'pqrs', '8':'tuv', '9':'wxyz'}
result = []
def letterCombinationsBacktrack(path, n, digits, idx):
    if idx == n:
        result.append(path)
        return
    
    for e in tel[digits[idx]]:
        path += e
        idx += 1
        letterCombinationsBacktrack(path, n, digits, idx)
        path = path[:-1]
        idx -= 1

def letterCombinations(digits):
    n = len(digits)
    if n == 0: return []
    path = ''
    idx = 0
    letterCombinationsBacktrack(path, n, digits, idx)
    return result

def xxx(dividend, divisor):
    if dividend == 0: return 0
    if (dividend > 0 and divisor > 0)\
        or (dividend < 0 and divisor < 0):
        sign = 1
    else: sign = -1
    dd, dr = abs(dividend), abs(divisor)
    if dr == 1:
        if sign == 1: return dd
        if sign == -1: return -dd
    if dd < dr: return 0

#sys.setrecursionlimit(100000)

def partition(A, p, r):
    i = p - 1
    x = A[r]
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]
    i += 1
    A[i], A[r] = A[r], A[i]
    return i

data = [(0,1),(1,1),(2,1),(3,-1),(4,-1),(5,-1),(6,1),(7,1),(8,1),(9,-1)]
def Adaboost(data, M):
    N = len(data)
    D = [1 / N] * N
    alpha = []
    best_threshold = []
    for m in range(M):
        min_e = float('inf')
        for threshold in np.linspace(0.5, 8.5, 9):
            e = 0
            for i in range(N):
                if data[i][0] < threshold: G = 1
                else: G = -1
                if data[i][1] != G:
                    e += D[i]
            min_e = min(e, min_e)

def trap(height):
    n = len(height)
    if n == 0: return 0
    ans = 0
    left = (0, 0)
    maybe = False
    for i in range(left[0], n):
        if not maybe:
            if height[i] == 0:
                continue
            else:
                left = (i, height[i])
                maybe = True
                continue
        if height[i] >= left[1]:
            for j in range(left[0] + 1, i):
                ans += (left[1] - height[j])
            left = (i, height[i])
    maybe = False
    for i in range(n - 1, left[0] - 1, -1):
        if not maybe:
            if height[i] == 0:
                continue
            else:
                left = (i , height[i])
                maybe = True
                continue
        if height[i] > left[1]:
            for j in range(left[0] - 1, i, -1):
                ans += (left[1] - height[j])
            left = (i, height[i])
    return ans

from datetime import datetime, timedelta
a = datetime(2020, 7, 23)
b = datetime(2019, 4, 11)
c = timedelta(days=28)
print(a - b)
fday = datetime(2016, 4, 25)
for i in range(4):
    lday = fday + timedelta(days=7)
    print(fday, lday)
    fday = lday

def f(wd, we):
    u_e, sigma_e = 0.13, 0.2
    u_d, sigma_d = 0.08, 0.12
    p = 0.3
    u_a = wd * u_d + we * u_e
    sigma_a = (wd**2)*(sigma_d**2)+(we**2)*(sigma_e**2)+2*p*wd*we*sigma_d*sigma_e
    print(u_a,sigma_a)
f(0.7,0.3)

min_cost, cost = float('inf'), 0
min_path, path = [], []
#数字与城市之间的映射
def city_name(num):
    idx = {0:'广州',1:'佛山',2:'深圳',3:'珠海',4:'韶关',5:'汕头',6:'湛江'}
    return idx[num]
#路费矩阵
mat = [[0,20,75,55,110,230,300],
       [20,0,80,45,130,250,270],
       [75,80,0,80,200,180,380],
       [55,45,80,0,370,390,280],
       [110,130,200,370,0,320,370],
       [230,250,180,390,320,0,550],
       [300,270,380,280,370,550,0]]
def back(citys, prev):
    global cost, min_cost, path, min_path
    n = len(citys)
    if n == 1:
        cost += (mat[prev][citys[0]] + mat[citys[0]][0])
        path.append(citys[0])
        if cost < min_cost:
            min_cost = cost
            min_path = path.copy()
        cost -= (mat[prev][citys[0]] + mat[citys[0]][0])
        path.pop()
        return
    for i in range(n):
        cost += mat[prev][citys[i]]
        path.append(citys[i])
        s = citys[:i] + citys[i+1:]
        back(s, citys[i])
        cost -= mat[prev][citys[i]]
        path.pop()
back([1,2,3,4,5,6], 0)
print('最低花费：', min_cost)
min_path = [0] + min_path + [0]
print('->'.join(map(city_name, min_path)))

def UF(board):
    m = len(board)
    if m == 0: return
    n = len(board[0])
    parent = [i for i in range(m * n + 1)]

    def find(node):
        while (parent[node] != node):
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    
    def union(node1, node2):
        root1, root2 = find(node1), find(node2)
        if root1 != root2:
            parent[root1] = root2
    
    def isConnected(node1, node2):
        return find(node1) == find(node2)
    
    def node(i, j):
        return i * n + j

    dummyNode = m * n
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'O':
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    union(node(i, j), dummyNode)
                else:
                    if i > 0 and board[i - 1][j] == 'O':
                        union(node(i, j), node(i - 1, j))
                    if i < m - 1 and board[i + 1][j] == 'O':
                        union(node(i, j), node(i + 1, j))
                    if j > 0 and board[i][j - 1] == 'O':
                        union(node(i, j), node(i, j - 1))
                    if i < n - 1 and board[i][j + 1] == 'O':
                        union(node(i, j), node(i, j + 1))
    
    for i in range(m):
        for j in range(n):
            if not isConnected(node(i, j), dummyNode):
                board[i][j] = 'X'

import torch
import random
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输⼊的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    print(example_indices)
    # 返回从pos开始的⻓为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
    # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')

def bub(r, n):
    flag = 1
    p1, p2 = 0, n - 2
    while flag and p2 > p1:
        flag = 0
        for j in range(p1, p2 + 1):
            if r[j] > r[j + 1]:
                flag = 1
                r[j], r[j + 1] = r[j + 1], r[j]
        for j in range(p2, p1, -1):
            if r[j] < r[j - 1]:
                flag = 1
                r[j], r[j - 1] = r[j - 1], r[j]
        p1 += 1
        p2 -= 1

def sort(r, n):
    for i in range(1, n):
        key = r[i]
        j = i - 1
        while j >= 0 and key[1] > r[j][1]:
            r[j + 1] = r[j]
            j -= 1
        r[j + 1] = key
    for i, j in enumerate(r):
        if i % 5 == 4: end = '\n'
        else: end = ' '
        print(j, end=end)

l = list(enumerate(range(10)))
sort(l,10)

import psutil
print(psutil.cpu_count())

import numpy as np
l = np.arange(1913).reshape(-1, 1)
x, y = [], []

for i in range(1913 - 28 - 1):
    _x = l[i:(i+28)]
    _y = l[i+28]
    x.append(_x)
    y.append(_y)
print(x[-1], y[-1])
print(l[-1, :])

l = [1,2,3]
l2 = [3,3,3]
l.extend(l2)
print(l)

import math
a = 1 * math.exp(0.05*4) * math.exp(0.065*1)
b = math.log(a)
print(b/5)

import pandas as pd 
d = {'金额':[800,200,300],
     '是否到期':['未到期','已到期','未到期']}
df = pd.DataFrame(d)
print(df.columns)

def takeSecond(ele):
    return ele[0]
l = [[31176229,84553602],[59484421,74029340],[8413784,65312321],[34575198,108169522],[49798315,88462685],[29566413,114369939],[12776091,37045071],[11759956,61001829],[37806862,80806032],[82906996,118404277]]
l.sort(key=takeSecond)#, reverse=True)
print(l)


result = []
step = [(1,2),(2,1),(2,-1),(1,-2)]
n, m = 9, 5
i, j = 1, 1
start = (1, 1)
path = [(start, 0)]
while path:
    pos, net = path.pop()
    for i in range(net, 4):
        next_pos = (pos[0] + step[i][0], pos[1] + step[i][1])
        if next_pos[0] == n and next_pos[1] == m:
            solution = [j[0] for j in path] + [pos] + [[9,5]]
            result.append(solution)
        elif (next_pos[0] <= n) and (1 <= next_pos[1] <= m):
            path.append((pos, i + 1))
            path.append((next_pos, 0))
            break
for solution in result:
    print(solution)

def isPrime(x):
    k = int(x**0.5 + 1)
    for i in range(2, k):
        if x % i == 0:
            return False
    return True

def get_rc(idx):
    return idx // 3, idx % 3

def get_idx(row, col):
    return 3 * row + col

result = []
path = []
nums = list(range(1, 11))
flag = [0] * 10
def backtrack(path):
    idx = len(path)
    
    if len(path) == 9:
        result.append(path.copy())

    for num in nums:
        if num == 1 and idx not in [0, 1, 4]: continue#去重
        if flag[num - 1] == 1: continue
        on = False
        for i, j in [(1,0),(0,-1),(-1,0),(-1,0)]:
            x, y = get_rc(idx)
            x2, y2 = x + i, y + j
            idx2 = get_idx(x2, y2)
            if 0 <= x2 <= 2 and 0 <= y2 <= 2 and idx2 < idx:
                if not isPrime(num + path[idx2]):
                    on = True#用于跳出外循环
        if on: continue
        path.append(num)
        flag[num - 1] = 1
        backtrack(path)
        path.pop()
        flag[num - 1] = 0

backtrack(path)
for solution in result:
    print(solution)
print(f'共{len(result)}个解')

def maxsum(nums):
    n = len(nums)
    if n == 0: return 0
    #if n == 1: return nums[0]
    s, b = -float('inf'), 0
    for i in range(n):
        if b > 0:
            b += nums[i]
        else:
            b = 0
            b += nums[i]
        if b > s:
            s = b
    return s

l = [-2, -1]
print(maxsum(l))

def takeSecond(ele):
    return ele[1]
l = [(0,2),(1,5),(2,3)]
l.sort(key=takeSecond, reverse=False)
print(l)
a = [1,2,3,4,5,6]
print(list(enumerate(a)))

def f(n, L, a):
    x = [0] * n
    a = list(enumerate(a))
    def takeSecond(ele):
        return ele[1]
    a.sort(key=takeSecond, reverse=False)
    for i in range(n):
        if a[i][1] <= L:
            x[a[i][0]] = 1
            L -= a[i][1]
        else:
            break

def backtrack(n, ans, solution, left, right):
    if len(solution) == n * 2:
        ans.append(solution)
        return       
    if left + 1 <= n:
        left += 1
        solution += '('
        backtrack(n, ans, solution, left, right)
        left -= 1
        solution = solution[:-1]
    if left > right:
        right += 1
        solution += ')'
        backtrack(n, ans, solution, left, right)
        right -= 1
        solution = solution[:-1]
def f(n):
    ans = []
    solution = ''
    backtrack(n, ans, solution, 0, 0)
    return ans

for sol in f(4):
    print(sol)

def f(table):
    m, n = len(table), len(table[0])
    dp = [[0] * n for _ in range(m)]
    ans = 0
    for i in range(m):
        dp[i][0] = table[i][0]
        ans = max(ans, dp[i][0])
    for j in range(1, n):
        dp[0][j] = table[0][j]
        ans = max(ans, dp[0][j])
    for i in range(1, m):
        for j in range(1, n):
            if table[i][j] == 0:
                continue
            if dp[i - 1][j] == 0 or dp[i][j - 1] == 0:
                dp[i][j] = 1
            elif dp[i - 1][j] != dp[i][j - 1]:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1
            else:
                l = dp[i - 1][j]
                if table[i - l][j - l] != 0:
                    dp[i][j] = l + 1
                else:
                    dp[i][j] = l
            ans = max(ans, dp[i][j])
    return ans ** 2
#table = [[0,0,0,1],[1,1,0,1],[1,1,1,1],[0,1,1,1],[0,1,1,1]]
table = [[1,0,1,0,0],[1,0,1,1,1],[1,1,1,1,1],[1,0,0,1,0]]
print(f(table))

rf = OneVsRestClassifier(RandomForestClassifier())
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
y_prob = rf.predict_proba(x_test)

ac = (y_test == y_pred).sum(axis=0) / y_test.shape[0]
ll = log_loss(y_test, y_prob) / 24
hl = hamming_loss(y_test, y_pred)
print('Accuracy:', ac.mean())
print('Log loss:', ll)
print('Hamming loss:', hl)

def BlackLitterman(df, tau, P, Q):
    mu = df.mean()
    sigma = df.cov()
    Pi = np.expand_dims(mu,axis = 0).T
    ts = tau * sigma
    ts_inv = linalg.inv(ts)
    Omega = np.dot(np.dot(P, ts), P.T) * np.eye(Q.shape[0])
    Omega_inv = linalg.inv(Omega)
    er = np.dot(linalg.inv(ts_inv + np.dot(np.dot(P.T, Omega_inv), P)),\
        (np.dot(ts_inv, Pi) + np.dot(np.dot(P.T, Omega_inv), Q)))
    posterirorSigma = linalg.inv(ts_inv + np.dot(np.dot(P.T, Omega_inv), P))
    return [er, posterirorSigma]

import time
import torch
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features = 512 * 7 * 7
fc_hidden_units = 4096

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_' + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module('fc', nn.Sequential(FlattenLayer(),
                            nn.Linear(fc_features, fc_hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(fc_hidden_units, fc_hidden_units),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(fc_hidden_units, 10)))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape:', X.shape)

def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i * i
maybe = createGenerator()
test = next(maybe)
print(test)
print(maybe)
for i in maybe:
    print(i)
for i in maybe:
    print(i)

from colorama import Fore, Back, Style
print(Fore.RED + 'some red text')
print(Back.GREEN + 'and with a green background')
print(Style.DIM + 'and in dim text')
print('hello')
print(Style.RESET_ALL)
print('back to normal now')

import os
path = 'D:/eat/code/mypy/mPi/data'

for root, dirs, files in os.walk(path):
    print(root, dirs, files)
    print('*' * 60)
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))

l = list(range(10))
new_l = filter(lambda x: x % 2 == 0, l)
for ele in new_l:
    print(ele)
for ele in new_l:
    print(ele)

from scipy.optimize import minimize
import numpy as np

def fun(args):
    a=args
    #v=lambda x:a/x[0] +x[0]
    v=lambda x: x**2 + 2*x + 1
    return v

args = (1)  #a
x0 = np.asarray((2))  # 初始猜测值
res = minimize(fun(args), x0, method='SLSQP')
print(type(res))
print(res.fun)
print(res.success)
print(res['x'])

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, higher_better=False):

        if higher_better: score = val_loss
        else: score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def EditDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(1, n + 1):
        dp[0][i] = i
    for i in range(1, m + 1):
        dp[i][0] = i
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, )

s = {'wyz': 1, 'zxy': 3}
tmp = {key: value for key, value in s.items()}
print('wyz' in s)

path = 'D:/SYSU/c++.txt'
file = open(path)
lines = file.readlines()
file.close()
print(lines)

path = 'D:/SYSU/c++.txt'
for line in open(path).readline():
    print(line)

import time
def runtime(func):
    def get_time():
        #func()
        print(time.time())
        #func()
    return get_time
@runtime
def student_run():
    print('学生跑')

student_run()

import numpy as np
import matplotlib.pyplot as plt
position = 0
walk = [position]
steps = 10
for i in range(steps):
    step = 1 if np.random.randint(0, 2) else -1
    position += step
    walk.append(position)
print((np.abs(walk) > 1))
print((np.abs(walk) > 1).argmax())

print(1E3)#1000.0


test = '面试官您好，我叫王业智，是中山大学应用统计专业的一名研二学生.'
seps, strips = u'\n。！？!?；;，, ', u'；;，, '
def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]
print(text_segmentate(test, 3, seps, strips))

import re
from collections import Counter

test = "One week in 2007, two friends (Dean and Bill) independently told me they were amazed at Google's spelling correction."
maybe = re.findall('\w+', test.lower())
path = 'D:/SYSU/c++.txt'
file = open(path)
print(Counter(re.findall('\w+', file.read())))
file.close()


import requests
import urllib
keyword = '猫'
print(urllib.parse.quote(keyword))


import urllib.request
response = urllib.request.urlopen("https://baike.baidu.com")
html = response.read().decode('utf8')
print(type(html))
print(html)

import requests
html = requests.get('https://baike.baidu.com')
print(type(html))
print(html)

import requests
headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
html = requests.get('https://baike.baidu.com', headers=headers).text
print(type(html))
print(html)

print(round(1.2, 2))

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# n1 = ListNode(1)
# n2 = ListNode(2)
# n3 = n1
# n3.next = n2
# print(n1.next.val)

def List2Node(l):
    if len(l) == 0: return None
    head = ListNode(l[0])
    head_cp = head
    for ele in l[1:]:
        head_cp.next = ListNode(ele)
        head_cp = head_cp.next
    return head

def out(head):
    l = []
    while head:
        l.append(head.val)
        head = head.next
    print(l)

l1 = List2Node([-10,-10,-9,-4,1,6,6])
l2 = List2Node([-7])
def first_step(l1, l2):
    if l1.val > l2.val:
        l1, l2 = l2, l1
    head_node = l1
    insert_node = l1.next
    l1.next = l2
    last_node = l1
    next_node = l2
    return head_node, last_node, next_node, insert_node
def f(l1, l2):
    head_node, last_node, next_node, insert_node = first_step(l1, l2)
    while insert_node and next_node:
        if last_node.val <= insert_node.val <= next_node.val:
            next_insert_node = insert_node.next
            last_node.next = insert_node
            insert_node.next = next_node
            insert_node = next_insert_node
            tail_node = next_node
            last_node = last_node.next
        else:
            tail_node = next_node
            last_node = last_node.next
            next_node = next_node.next
    if insert_node: tail_node.next = insert_node
    return head_node
out(f(l1, l2))

def reverse(head):
    if head is None: return
    last_node = None
    while head.next:
        next_node = head.next
        head.next = last_node
        last_node = head
        head = next_node
    head.next = last_node
    return head

def reverse2(pHead):
    last_node = None
    while pHead:
        next_node = pHead.next
        pHead.next = last_node
        last_node = pHead
        pHead = next_node
    return last_node

def isPalindrome(head):
    # write code here
    if head is None: return True
    dummy = ListNode(None)
    dummy.next = head
    slow_p, fast_p = dummy, head
    while fast_p.next:
        slow_p = slow_p.next
        fast_p = fast_p.next
        if fast_p.next:
            fast_p = fast_p.next
    mid_node = slow_p.next
    slow_p.next = None
    new_head = reverse(mid_node)
    out(head)
    out(new_head)
    while head and new_head:
        if head.val != new_head.val:
            return False
        head = head.next
        new_head = new_head.next
    return True
head = List2Node([])
print(isPalindrome(head))

case = 1
if case == 3:
    print('hhh')
elif case == 2:
    print('xxx')#可以没有else

l = [1, 2]

from random import randint

def partition(arr):
    n = len(arr)
    m = randint(0, n - 1)
    arr[-1], arr[m] = arr[m], arr[-1]
    key = arr[-1]
    p = -1
    for i in range(n - 1):
        if arr[i] > key:
            p += 1
            arr[i], arr[p] = arr[p], arr[i]
    arr[p + 1], arr[-1] = arr[-1], arr[p + 1]
    return p + 2, arr[p + 1]

def f(a, n, K):
    while True:
        m, val = partition(a)
        if m == K:
            return val
        elif m > K:
            a = a[:m - 1]
        elif m < K:
            a = a[m:]
            K = K - m
#a = [1,3,5,2,2,2,3]
a, n, K = [1332802,1177178,1514891,871248,753214,123866,1615405,328656,1540395,968891,1884022,252932,1034406,1455178,821713,486232,860175,1896237,852300,566715,1285209,1845742,883142,259266,520911,1844960,218188,1528217,332380,261485,1111670,16920,1249664,1199799,1959818,1546744,1904944,51047,1176397,190970,48715,349690,673887,1648782,1010556,1165786,937247,986578,798663],49,24
print(f(a, n, K))

from copy import deepcopy
dp = [[1,2],[3,4]]
#dp[0] = dp[1]
#dp[0] = dp[1].copy()
dp[0] = deepcopy(dp[1])
dp[1][0] = 999
print(dp)

#union isconnect

import sys
a, b, c = sys.stdin.readline().strip().split()
print((a, b, c))

a = [1,2,3,4,5,6]
a[3:] = [9]*3
print(a)

def f(stones):
    m = len(stones)
    if m == 0: return 0
    s = sum(stones)
    n = s // 2
    dp = [[0] * (n + 1) for _ in range(m)]
    if stones[0] <= n:
        dp[0][stones[0]:] = [stones[0]] * (n - stones[0] + 1)
    for i in range(1, m):
        for j in range(1, n + 1):
            if j < stones[i]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-stones[i]]+stones[i])
    return abs(s - 2 * dp[-1][-1])
    #return dp[-1][-1]
stones = [31,26,33,21,40]
maybe = f(stones)
print(maybe)
for ele in maybe:
    print(ele)
print(len(maybe), len(maybe[0]))

from copy import deepcopy
dp = [[1,2,3],[4,5,6]]
dp2 = deepcopy(dp)
#dp2 = dp.copy()
dp2[0][0] = 9
dp2[1][1] = 9
print(dp)

def f(matrix):
    m = len(matrix)
    if m == 0: return 0
    n = len(matrix[0])
    if n == 0: return 0
    h = [[0] * n for _ in range(m)]
    w = [[0] * n for _ in range(m)]
    ans = 0
    if matrix[0][0] == '1':
        h[0][0], w[0][0] = 1, 1
        ans = 1
    for i in range(1, n):
        if matrix[0][i] == '1':
            h[0][i] = 1
            w[0][i] = w[0][i-1] + 1
            ans = max(ans, w[0][i] * h[0][i])
    for i in range(1, m):
        if matrix[i][0] == '1':
            h[i][0] = h[i-1][0] + 1
            w[i][0] = 1
            ans = max(ans, w[i][0] * h[i][0])
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == '1':
                if w[i][j-1] <= w[i-1][j] - 1:
                    w[i][j] = w[i][j-1] + 1
                    h[i][j] = h[i-1][j] + 1
                    ans = max(ans, w[i][j] * h[i][j])
                else:
                    w[i][j] = w[i][j - 1] + 1
                    if h[i][j-1] - 1 >= h[i-1][j]:
                        h[i][j] = 1 + h[i-1][j]
                    else:
                        h[i][j] = h[i][j-1]
                    ans = max(ans, w[i][j] * h[i][j])
    return h, w
    #return ans
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
#print(f(matrix))
h, w = f(matrix)
for i, j, k in zip(matrix, h, w):
    print(i, j, k)

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        from queue import Queue
        m = len(board)
        if m == 0: return
        n = len(board[0])
        q = Queue()
        for i in range(n):
            if board[0][i] == 'O':
                board[0][i] = 'B'
                q.put((0, i))
            if board[m - 1][i] == 'O':
                board[m - 1][i] = 'B'
                q.put((m - 1, i))
        for i in range(1, m - 1):            
            if board[i][0] == 'O':
                board[i][0] = 'B'
                q.put((i, 0))
            if board[i][n - 1] == 'O':
                board[i][n - 1] = 'B'
                q.put((i, n - 1))
        while not q.empty():
            x, y = q.get()
            for i, j in [(1,0),(-1,0),(0,1),(0,-1)]:
                a, b = x + i, y + j
                if 0 <= a <= m - 1 and 0 <= b <= n - 1:
                    if board[a][b] == 'O':
                        board[a][b] = 'B'
                        q.put((a, b))
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == 'B':
                    board[i][j] = 'O'

from queue import Queue
q = Queue()
q.put((1, 2, None))
print(q.get())

if 'h': print('hhh')

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def parent(i):
    return int((i-1)/2)
def left(i):
    return 2*i+1
def right(i):
    return 2*i+2
def list2tree(nums):
    n = len(nums)
    nodes = []
    for i in range(n):
        if nums[i] is not None:
            nodes.append(TreeNode(nums[i]))
        else:
            nodes.append(None)
    for i in range(n):
        if nodes[i] is not None:
            if 0 <= left(i) < n and nodes[left(i)] is not None:
                if nodes[left(i)] is not None:
                    nodes[i].left = nodes[left(i)]
            if 0 <= right(i) < n and nodes[right(i)] is not None:
                if nodes[right(i)] is not None:
                    nodes[i].right = nodes[right(i)]
    return nodes[0]
ans = []
def out(root):
    if root.left is not None:
        out(root.left)
    ans.append(root.val)
    if root.right is not None:
        out(root.right)
# 树节点列表
# for ele in nodes:
#     if ele is not None:
#         print('root:', ele.val)
#     else:
#         continue
#     if ele.left is not None:
#         print(f'{ele.val}\'s left:', ele.left.val)
#     if ele.right is not None:
#         print(f'{ele.val}\'s right:', ele.right.val)
def f(root):
    if root is None: return True
    from queue import Queue
    q = Queue()
    q.put((root, -float('inf'), float('inf')))
    while not q.empty():
        node, lower, upper = q.get()
        if not (lower < node.val < upper):
            return False
        if node.left is not None:
            if node.left.val >= node.val:
                return False
            q.put((node.left, lower, node.val))
        if node.right is not None:
            if node.right.val <= node.val:
                return False
            q.put((node.right, node.val, upper))
    return True
null = None
nums = [2,1,3]
nodes = list2tree(nums)
print(f(nodes))

for i in range(2, 1):
    print('h')

print(ord('a'), '->', ord('z'))
print(ord('A'), '->', ord('Z'))
print(ord('z') - ord('Z'))
print(chr(97), '->', chr(122))
for j in range(97, 123): print(chr(j), end='')

def f(nums):
    n = len(nums)
    dp1, dp2 = [0] * n, [0] * n
    dp1[0] = dp2[0] = ans = nums[0]
    for i in range(1, n):
        dp1[i] = max(nums[i], nums[i]*dp1[i-1], nums[i]*dp2[i-1])
        dp2[i] = min(nums[i], nums[i]*dp1[i-1], nums[i]*dp2[i-1])
        ans = max(ans, dp1[i])
    return ans
print(f([-2]))
print(len([[]]))

from copy import deepcopy
dp = [[1,2],[3,4]]
#dp2 = dp
#dp2 = dp.copy()
dp2 = deepcopy(dp)
dp2[0][0] = 999
print(dp)

a = 1
def f():
    global a
    a += 1
    return a
a = f()
print(a)

def f():
    global area
    area += 1
area = 1
f()

def f(grid):
    def dfs(i, j):
        global area
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < m and 0 <= new_j < n:
                if grid[new_i][new_j] == 1:
                    grid[new_i][new_j] = 0
                    area += 1
                    dfs(new_i, new_j)

    m = len(grid)
    if m == 0: return 0
    n = len(grid[0])
    if n == 0: return 0
    ans = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                grid[i][j] = 0
                global area
                area = 1
                dfs(i, j)
                ans = max(ans, area)

    return ans
grid = [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]

print(f(grid))

strs = ["10", "0001", "111001", "1", "0"]
l = list(map(lambda s: (s.count('0'), s.count('1')), strs))
print(l)


def f(strs, m, n):
    s = len(strs)
    strs = list(map(lambda s: (s.count('0'), s.count('1')), strs))
    dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(s)]
    for i in range(m + 1):
        for j in range(n + 1):
            if strs[0][0] <= i and strs[0][1] <= j:
                dp[0][i][j] = 1
    for i in range(1, s):
        for j in range(m + 1):
            for k in range(n + 1):
                new_j, new_k = j - strs[i][0], k - strs[i][1]
                if new_j >= 0 and new_k >= 0:
                    dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][new_j][new_k] + 1)
                else:
                    dp[i][j][k] = dp[i-1][j][k]
    return dp[-1][-1][-1]
strs = ["10", "0001", "111001", "1", "0"]
m, n = 5, 3
print(f(strs, m, n))


print(int('01'))
print(int('11', 2))

a = '123'#[1,2,3]
b = a
#del a
a = a[:-1]
#a.pop()
print(b)

a = ''
a += '1'
print(a)

from heapq import heapify, heappop, heappush
nums = [2,7,4,1,8,1]
#nums = []
heapify(nums)
print(nums[-1])
print(heappop(nums))
print(len(nums))
print(heappop(nums))
print(nums)
if nums: print('hhh')

print('Hello world from process %d at %s.' % (rank, node_name))
import logging
print(logging.__version__)
logging.debug(u"苍井空")
logging.info(u"麻生希")
logging.warning(u"小泽玛利亚")
logging.error(u"桃谷绘里香")
logging.critical(u"泷泽萝拉")
test = [['w','u','z'],['a','b','c']]
l = [1,2,3]
print(list(reversed(l)))
exit()
#flatten = lambda l: [item for sublist in l for item in sublist]
#flatten = lambda l: [item for item in sublist for sublist in l]
print(flatten(test))

from collections import defaultdict
d = defaultdict(bool)

print(ord('A'), ord('Z'), ord('a'), ord('z'))

print(ord('0'), ord('1'), ord('9'))

def f(s):
    def isstr(s):
        return 65 <= ord(s) <= 90 or 97 <= ord(s) <= 122
    def isdigit(s):
        return 48 <= ord(s) <= 57

    ans = ''
    n = len(s)
    stack = []
    i = 0
    #while i <= n-1:
    while i <= n-1:
        if isstr(s[i]):
            ans += s[i]
            i += 1
        elif isdigit(s[i]):
            tmp = s[i]
            i += 1
            while s[i] != '[':
                tmp += s[i]
                i += 1
            i += 1
            stack.append([int(tmp), ''])
            while stack and i <= n-1:
                if isstr(s[i]):
                    stack[-1][1] += s[i]
                    i += 1
                elif isdigit(s[i]):
                    tmp = s[i]
                    i += 1
                    while s[i] != '[':
                        tmp += s[i]
                        i += 1
                    i += 1
                    stack.append([int(tmp), ''])
                elif s[i] == ']':
                    if len(stack) == 1:
                        node = stack.pop()
                        ans += node[0] * node[1]
                    else:
                        node = stack.pop()
                        stack[-1][1] += node[0] * node[1]
                    i += 1
                elif s[i] == '[':
                    i += 1
    return ans
s = "3[a2[c]]"
print(f(s))

l = [[1, '']]
l[0][1] += 'w'
print(l)

if None:
    print('hhh')
d = {-6:False, -10:False, -15:False}
if -6 in d.keys(): print('hhh')

from heapq import heapify, heappop, heappush
maybe = [9,1,2,3,4,5]
heapify(maybe)
print(maybe)
def f(n):
    from queue import Queue
    q = Queue()
    q.put(-1)
    from heapq import heappop, heapify, heappush
    ans = []
    heapify(ans)
    prev = None
    d = {-6:False, -10:False, -15:False}
    while not (len(ans) >= n and ans[0] != prev):
        val = q.get()
        for i in [2, 3, 5]:
            tmp = val * i
            if tmp in d.keys():
                if d[tmp]:
                    continue
                d[tmp] = True
            q.put(tmp)
        if len(ans) != 0:
            prev = ans[0]
        heappush(ans, val)
    for _ in range(len(ans)-n):
        heappop(ans)
    return -ans[0]
    #return ans
print(f(7))
for i in range(7, 13):
    print(f(i))


p = [1,2,3,4,5]
a = p.copy()
a[0] = 9
print(p)

a = None
a = set()
a.add(1)
print(a)

def sumOfDistancesInTree(self, N: int, edges: List[List[int]]) -> List[int]:
    graph = [[] for _ in range(N)]
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    dist_sum = [0 for _ in range(N)]
    node_num = [1 for _ in range(N)]

    def post_order(node, parent):
        for n in graph[node]:
            if n == parent:
                continue
            post_order(n, node)
            node_num[node] += node_num[n]
            dist_sum[node] += dist_sum[n] + node_num[n]

    def pre_order(node, parent):
        for n in graph[node]:
            if n == parent:
                continue
            dist_sum[n] = dist_sum[node] - node_num[n] + (N - node_num[n])
            pre_order(n, node)
    
    post_order(0, -1)
    pre_order(0, -1)
    return dist_sum

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

root = TreeNode(-2)
root.right = TreeNode(-3)
def f(root, sum):
    ans = []
    if root is None:# or root.val > sum:
        return ans
    path = [root.val]
    s = root.val
    def dfs(node, s, path):
        # global path
        if node.left is None and node.right is None:
            if s == sum:
                ans.append(path.copy())
            return
        if node.left is not None:
            path.append(node.left.val)
            s += path[-1]
            # if s > sum:
            #     s -= path[-1]
            #     path.pop()
            # else:
            dfs(node.left, s, path)
            s -= path[-1]
            path.pop()
        if node.right is not None:
            path.append(node.right.val)
            s += path[-1]
            # if s > sum:
            #     s -= path[-1]
            #     path.pop()
            # else:
            dfs(node.right, s, path)
            s -= path[-1]
            path.pop()
    dfs(root, s, path)
    return ans

print(f(root, -5))

l = [1, 1]
l.append(l.pop() + l.pop())
print(l)
print(ord('a'), ord('z'))
print(list('aaa'))

def f(A):
    n = len(A)
    from collections import defaultdict
    dicts = [defaultdict(int) for _ in range(n)]
    for i in range(n):
        for char in A[i]:
            dicts[i][char] += 1
    ans = []
    for i in range(97, 123):
        def f(d):
            return d[chr(i)]
        tmp = chr(i) * min(map(f, dicts))
        ans += list(tmp)
    return ans
a, b = ["bella","label","roller"], ["cool","lock","cook"]
print(f(b))
print(1 and 0)
l1 = [1, 2, 3]
l2 = [2,3,4]
l1 = l2.copy()
l2[0] = 999
print(l1)

def f(nums):
    s = sum(nums)
    n = len(nums)
    if s % 2 == 1 or n == 1: return False
    target = s // 2
    dp1 = [0] * (target + 1)
    dp1[0] = 1
    if nums[0] <= target:
        dp1[nums[0]] = 1
    dp2 = [1] * (target + 1)
    for i in range(1, n):
        for j in range(1, target + 1):
            if j - nums[i] < 0:
                dp2[j] = dp1[j]
            else:
                dp2[j] = dp1[j] or dp1[j - nums[i]]
        dp1 = dp2.copy()
    return bool(dp2[-1])

nums = [100]
print(f(nums))

from queue import Queue
q = Queue()
print('hh')
q.put(3)
node = q.get()
print(node)

a = (1)
b = (1)
print(a is b)

def f(s):
    def isNum(x):
        return 48 <= ord(x) <= 57
    def compute(comp, a, b):
        if comp == '+': return a + b
        elif comp == '-': return a - b
    n = len(s)
    i = 0
    stack = []
    while i < n:
        if s[i] == ' ':
            i += 1
        elif s[i] in ['(', '+', '-']:
            stack.append(s[i])
            i += 1
        elif isNum(s[i]):
            start = i
            i += 1
            while i < n and isNum(s[i]):
                i += 1
            if (not stack) or stack[-1] == '(':
                stack.append(int(s[start:i]))
                continue
            if stack[-1] in ['+', '-']:
                cache = compute(stack.pop(), stack.pop(), int(s[start:i]))
                stack.append(cache)
        elif s[i] == ')':
            cache = stack.pop()
            stack.pop()
            if not stack:
                stack.append(cache)
            elif stack[-1] in ['+', '-']:
                cache = compute(stack.pop(), stack.pop(), cache)
                stack.append(cache)
            i += 1
    return stack[-1]

s = "(1+(4+5+2)-3)+(6+8)"
print(f(s))

l = [3,1,5,2]
l.sort()
print(l)

def f(intervals):
    def f(ele):
        return ele[1]
    intervals.sort(key=f)
    # ans = 0
    n = len(intervals)
    if n <= 1: return 0
    ans = 0
    prev = intervals[0]
    for i in range(1, n):
        if intervals[i][0] >= prev[1]:
            prev = intervals[i]
        else:
            ans += 1
    return ans
i = [ [1,2], [1,2], [1,2] ]
print(f(i))

def f(x):
    pass
print(type(f))

class A(object):
    bar = 1
a = A()
print(getattr(a, 'bar'))

l = [3,2,1]
def f(nums):
    nums[:] = nums[::-1]
f(l)
print(l)

1 5 8 4 7 6 5 3 1

from copy import deepcopy
a = [[1,2],[3,4]]
b = deepcopy(a)
b[1][1] = 9
print(a)

def f(nunms, limit):
    n = len(nums)
    dp_max = [[0] * n for _ in range(n)]
    for i in range(n):
        dp_max[i][i] = nums[i]
    from copy import deepcopy
    dp_min = deepcopy(dp_max)
    for i in range(n):
        cache = nums[i]
        for j in range(i+1, n):
            cache = dp_max[i][j] = max(cache, nums[j])
            # cache = dp_max[i][j]
    for i in range(n):
        cache = nums[i]
        for j in range(i+1, n):
            cache = dp_min[i][j] = min(cache, nums[j])
    ans = 1
    for i in range(n):
        for j in range(i+1, n):
            if dp_max[i][j] - dp_min[i][j] <= limit:
                ans = max(ans, j-i+1)
    return ans
nums = [4,2,2,2,4,4,2,2]
limit = 0
print(f(nums, limit))

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        # 栈底维护了最值，栈顶维护了窗口终点

        ascend, descend = [], [] # 维护一个最小值栈，一个最大值栈
        a, d = 0, 0 # 队首下标
        t = 0 # 窗口起始点
        res = 0

        for i in range(len(nums)): # i为窗口终止点，利用循环滑动终止点
            temp = nums[i] # 保存终止点对应的值

            # 更新最小值
            while len(ascend) > a and temp < ascend[-1][0]: # 与栈顶值对比
                ascend.pop() # 弹出
            ascend.append((temp, i)) # 记录终点值与下标
            
            # 更新最大值
            while len(descend) > d and temp > descend[-1][0]: # 与栈顶值对比
                descend.pop() # 弹出
            descend.append((temp, i)) # 记录终点值与下标

            while descend[d][0] - ascend[a][0] > limit: # 最值之差大于limit出发滑动条件
                t += 1 # 起始点滑动
                if ascend[a][1] < t: # 如果窗口起点在最小值索引右侧，需要舍弃此时的最小值
                    a += 1
                if descend[d][1] < t: # 如果窗口起点在最大值索引右侧，需要舍弃此时的最大值
                    d += 1
            
            res = max(res, i-t+1) # 更新最大长度

        return res

from aliyunsdkcore import client
from aliyunsdksts.request.v20150401 import AssumeRoleRequest

clt = client.AcsClient('', '', '')

# 构造"AssumeRole"请求
request = AssumeRoleRequest.AssumeRoleRequest()

# 指定角色
request.set_RoleArn('')

# 设置会话名称，审计服务使用此名称区分调用者
request.set_RoleSessionName('maybe')

#request.set_method('HMAC-SHA1')

# 发起请求，并得到response
response = clt.do_action_with_exception(request)
print(response)


from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':
        # base cases
        n = len(nums)
        if n * k == 0:
            return []
        if k == 1:
            return nums
        
        def clean_deque(i):
            # remove indexes of elements not from sliding window
            if deq and deq[0] == i - k:
                deq.popleft()
                
            # remove from deq indexes of all elements 
            # which are smaller than current element nums[i]
            while deq and nums[i] > nums[deq[-1]]:
                deq.pop()
        
        # init deque and output
        deq = deque()
        max_idx = 0
        for i in range(k):
            clean_deque(i)
            deq.append(i)
            # compute max in nums[:k]
            if nums[i] > nums[max_idx]:
                max_idx = i
        output = [nums[max_idx]]
        
        # build output
        for i in range(k, n):
            clean_deque(i)          
            deq.append(i)
            output.append(nums[deq[0]])
        return output


from collections import deque
q = deque()
print(len(q))
if q: print('hhh')

import torch
box1 = torch.tensor([[0, 0, 3, 3], [0, 1, 3, 3]])
box2 = torch.tensor([[1, 1, 4, 4], [2, 0, 4, 2]])
def IoU(box1, box2):
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(  # 左上角的点
        box1[:, :2].unsqueeze(1).expand(N, M, 2),   # [N,2]->[N,1,2]->[N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),   # [M,2]->[1,M,2]->[N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0   # 两个box没有重叠区域
    inter = wh[:,:,0] * wh[:,:,1]   # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # (N,)
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # (M,)
    area1 = area1.unsqueeze(1).expand(N,M)  # (N,M)
    area2 = area2.unsqueeze(0).expand(N,M)  # (N,M)

    iou = inter / (area1+area2-inter)
    return iou
def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor

s = "  Bob    Loves  Alice   "
def f(s):
    s = s.strip().split()
    return ' '.join(s[::-1])
print(f(s))


def g(arr1, arr2):
    n1 = len(arr1)
    n2 = len(arr2)
    order = dict()
    for i in range(n2):
        order[arr2[i]] = i + 1
    arr3, arr4 = [], []
    for ele in arr1:
        if order.get(ele) is not None:
            arr3.append(ele)
        else:
            arr4.append(ele)
    def f(ele):
        return order[ele]
    arr3.sort(key=f, reverse=False)
    arr4.sort()
    return arr3 + arr4
arr1 = [2,3,1,3,2,4,6,7,9,2,19]
arr2 = [2,1,4,3,9,6]
print(g(arr1, arr2))

def f(customers, grumpy, X):
    def windows(i):
        ans = 0
        for j in range(X):
            ans += customers[i+j] * grumpy[i+j]
        return ans
    n = len(customers)
    if n <= X:
        return sum(customers)
    ans = 0
    for i in range(n):
        if grumpy[i] == 0:
            ans += customers[i]
    p = 0
    delta = 0
    while p <= n-X:
        if grumpy[p] == 1:
            delta = max(delta, windows(p))
        p += 1
    delta = max(delta, windows(p-1))
    return ans + delta
a = [6,10,2,1,7,9]
b = [1,0,0,0,0,1]
c = 3
print(f(a, b, c))

def f(s, t, maxCost):
    def getCost(s1, s2):
        return abs(ord(s1) - ord(s2))    

    n = len(s)
    if n == 0: return 0

    p = 0
    while p < n:
        if getCost(s[p], t[p]) <= maxCost:
            break
        p += 1
    if p == n: return 0
    left, right = p, p
    cur_cost = getCost(s[p], t[p])
    cur_length = 1
    ans = 1
    while right < n-1:
        right += 1
        cur_length += 1
        cur_cost = cur_cost + getCost(s[right], t[right])
        while cur_cost > maxCost:
            cur_cost -= getCost(s[left], t[left])
            cur_length -= 1
            left += 1
        ans = max(ans, cur_length)
    return ans


a = "abcd"
b = "abcd"
c = 0
print(f(a,b,c))

l = [1,2,3]
l += l
print(l)

# 同向 保鲜 理解 改变
def f(gas, cost):
    def findStart(start):
        if start >= n:
            return n
        while start < n:
            if gas[start] >= cost[start]:
                break
            start += 1
        return start

    n = len(gas)
    gas += gas
    cost += cost

    start = 0
    start = findStart(start)
    if start == n: return -1

    res = gas[start] - cost[start]
    end = start + 1
    while True:
        res = gas[end] + res - cost[end]
        if res < 0:
            start = findStart(end)
            if start == n: return -1
            res = gas[start] - cost[start]
            end = start + 1
        else:
            if end - start == n:
                return start
            end += 1
gas  = [3,3,4]
cost = [3,4,4]
print(f(gas, cost))

left, right = 0, 3
print((left + right) // 2)

def f(nums, target):
    def getMid(left, right):
        if left <= right: return left + (right - left) // 2
        else:
            maybe = left + (right + n - left) // 2
            if maybe >= n: return maybe - n
            else: return maybe

    n = len(nums)
    if n == 1:
        if nums[0] == target: return 0
        else: return -1
    right = None
    for i in range(n-1):
        if nums[i] > nums[i+1]:
            right = i
            break
    if right is None: right = n-1
    if right == n-1: left = 0
    else: left = right + 1
    if target < nums[left] or target > nums[right]: return -1

    # while left != right:
    while nums[left] <= nums[right]:
        mid = getMid(left, right)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            if mid == n-1: left = 0
            else: left = mid + 1
        elif nums[mid] > target:
            if mid == 0: right = n-1
            else: right = mid - 1
        # if target < nums[left] or target > nums[right]: return -1
    if nums[left] == target: return left
    return -1
nums = [242,245,249,250,252,253,257,262,263,268,275,280,282,283,285,290,292,293,297,299,4,5,8,9,10,14,16,17,18,20,22,23,29,32,36,39,47,51,56,68,70,73,75,77,79,81,82,89,98,100,107,108,112,114,115,117,118,119,128,131,134,139,142,147,148,154,161,162,165,167,171,172,174,177,180,183,189,190,191,192,194,197,200,203,206,207,208,209,210,212,217,220,223,226,227,231,237]
# print(nums[20], nums[37], nums[38])
target = 54
print(f(nums, target))

print(round(1.75*1.75*18.5, 2), round(1.75*1.75*23.9, 2))

def f(points):
    def takeFirst(x):
        return x[0]
    n = len(points)
    if n == 0 or n == 1: return n
    points.sort(key=takeFirst)
    p = 1
    cur = points[0]
    ans = 0
    while p < n:
        if cur[0] == points[p][0]:
            cur = [cur[0], min(cur[1], points[p][1])]
            p += 1
        elif cur[1] < points[p][0]:
            ans += 1
            cur = points[p]
            p += 1
        else:
            cur = [points[p][0], min(cur[1], points[p][1])]
            p += 1
    return ans + 1
l = [[31176229,84553602],[59484421,74029340],[8413784,65312321],[34575198,108169522],[49798315,88462685],[29566413,114369939],[12776091,37045071],[11759956,61001829],[37806862,80806032],[82906996,118404277]]
print(f(l))

print(not False)

class Node:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.nxt = None
        self.prv = None

class LinkedList:
    def __init__(self):
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.size = 0
        self.head.nxt = self.tail
        self.tail.prv = self.head

    def addLast(self, node):
        node.prv = self.tail.prv
        node.nxt = self.tail
        self.tail.prv.nxt = node
        self.tail.prv = node
        self.size += 1
    
    def remove(self, node):
        node.prv.nxt = node.nxt
        node.nxt.prv = node.prv
        self.size -= 1
    
    def removeFirst(self):
        if self.head.nxt == self.tail:
            return -1
        first_node = self.head.nxt
        self.remove(first_node)
        return first_node

class LRUCache:

    def __init__(self, capacity: int):
        self.hashmap = dict()
        self.cache = LinkedList()
        self.capacity = capacity

    def makeRecently(self, key):
        node = self.hashmap[key]
        self.cache.remove(node)
        self.cache.addLast(node)

    def addRecently(self, key, val):
        node = Node(key, val)
        self.cache.addLast(node)
        self.hashmap[key] = node
    
    def deleteKey(self, key):
        node = self.hashmap[key]
        self.cache.remove(node)
        self.hashmap.pop(key)
    
    def removeLeastRecently(self):
        node = self.cache.removeFirst()
        key = node.key
        self.hashmap.pop(key)

    def get(self, key: int) -> int:
        if self.hashmap.get(key) is None:
            return -1
        self.makeRecently(key)
        return self.hashmap[key].val

    def put(self, key: int, value: int) -> None:
        if self.hashmap.get(key) is not None:
            self.deleteKey(key)
            self.addRecently(key, value)
            return 
        if self.capacity == self.cache.size:
            self.removeLeastRecently()
        self.addRecently(key, value)

d = {'wyz':0, 'dd':1}
l = [(key, val) for key, val in d.items()]
print(l[0][1])

ans = [[1] * i for i in range(1, 4)]
print(ans)

from collections import Counter
l = Counter()
print(l, type(l))
l = [1,2,3,4,4,5]
print(Counter(l).values())

def change(c):
    num = ord(c)
    if 97 <= num <= 122:
        return chr(num - 32)
    else:
        return chr(num + 32)
print(change('a'), change('A'))

S = '123'
# S[0] = 'c'
print(type(''.join(list(S))))

a, b = divmod(10, 3)
l = list(range(1, 11))
for i in range(a):
    if i < b:

print(a, b)

print([None for _ in range(3)])
print(int('1111', 2))
print(4 * 2 ** 3)

import torchvision

print(torchvision.__version__)
print(torchvision.__path__) 

S = '12345'
print(S[1:999])

def f(S):
    n = len(S)
    for i in range(1, n):
        for j in range(1, n-i+1):
            fst = int(S[:i])
            scd = int(S[i:i+j])
            p = i+j
            ans = [fst, scd]
            while p < n:                    
                trd = fst + scd
                m = len(str(trd))
                tgt = int(S[p:p+m])
                if tgt != trd:
                    break
                else:
                    ans.append(trd)
                    fst = scd
                    scd = trd
                    p = p + m
                    if p == n: return ans
    return []
S = "11235813"
print(f(S))

print(ord('3') - ord('0'))

from functools import reduce
print(reduce(lambda x, y : x*y, [1,2,3,4,5]))

print(-1 % 256)