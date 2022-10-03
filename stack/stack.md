# 栈
栈在数据结构中的作用远比我过去以为的重要，主要可以分为四类问题。第一类问题是利用栈本身，第二类是辅助栈，第三类是双栈，最巧妙的应用还是第四类单调栈。
## 一、栈的特性
利用栈先入后出特性的题。在tree.md中我们介绍到递归是利用了系统栈进行分治操作，事实上我们可以用stack来自己写栈替换系统栈。所以我们会发现本节很多题会有两种解法，一种递归解法，一种利用栈来解题。这也验证了这两种写法的等价性，我们完全可以选择一种好理解的方式来掌握这种题。

[有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)
```python
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}',  '[': ']', '(': ')','?':'?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False 
        return len(stack) == 1
ss=Solution()
print(ss.isValid("]"))
```
本题要注意添加‘？’是用来防止程序报错的。stack.pop意味着stack不能是空的，如果是空的会报错。dit[]意味着dic中必须有'?'这个key，不然也会报错。整个程序的逻辑是只要字符串s中出现了非括号元素一定报错；当出现左括号就入栈，出现对应右括号就出栈。最后考虑边界问题:"?"是不是会导致出现bug的情况出现，事实上并不会，这是一个巧妙的设计。由于"?"的key和vaue相同，并不能进入出栈的代码，因此没有导致bug。本题是一个很具有典型性的模版，有很强的指导意义。
下面这种解法更加容易想到，但是由于要检查所有的值，没有上面的好。
```python
class Solution:
    def isValid(self, s: str) -> bool:
        dic={')':'(',']':'[','}':'{','?':'?'}
        stack=['?']
        for i in s:
            if i in dic and stack[-1]==dic[i]:
                stack.pop()
            else:
                stack.append(i)
        return len(stack)==1


ss=Solution()
print(ss.isValid("[]?"))

```

## 辅助栈
[最小栈](https://leetcode-cn.com/problems/min-stack/)
没有做过这个题的话肯定是一脸懵逼，不知道这个题的目的是什么意思。这个题目的核心目的是要有一个栈，其末尾记录的是其栈中的最小值。如果是这样一个目的，我们只要将[value,当前最小值]记录下来即可。
```python
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.append([x,x])
        else:
            self.stack.append([x,min(self.stack[-1][1],x)])

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```
[字符串解码](https://leetcode-cn.com/problems/decode-string/)
```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack,multi,res=[],0,''
        for i in s:
            if i=='[':
                stack.append([multi,res])
                multi,res=0,''
            elif i==']':
                pre_multi,pre_res = stack.pop()
                res = pre_res+pre_multi*res
            elif '0'<=i<='9':
                multi= 10*multi+int(i)
            else:
                res+=i
        return res
```
## 单调栈
单调栈最为关键的是明白什么时候出栈，什么时候入栈
[739. 每日温度](https://leetcode-cn.com/problems/daily-temperatures/)[Li]
这道题构造了一个单调减的栈，去完成由于时序不一致造成的问题。
```python
class Solution:
    def dailyTemperatures(self, T) :
        res = [0 for _ in T]
        stack = []
        for i in range(len(T)):
            while stack and T[i]>T[stack[-1]]:
                topIndex = stack.pop()
                res[topIndex]=i-topIndex
            stack.append(i)
        return res
```
[84 柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)[Ye]
本题作为单调栈题目，关键的是明白什么时候出栈，什么时候入栈。由于只给了一个数组，我们想到可以利用的特性只有递增或者递减。我们观察到如果数组呈现递增趋势，我们并不能确认确定最大的大小是多少，而如果呈现除了递减的情况，我们可以找到前面比递减数大的所有数来计算最大面积。由于此时并不一定能找到最大的面积，我们需要在最后加上一个0作为后哨。而由于边界问题，我们也要添加一个前哨。递增出栈还有一个很反常识的地方，保证栈中的值是递增的，这有一个什么好处呢？计算的时候，可以直接i - stack[-1] - 1) * heights[tmp]，由于栈递增，直接算面积就不用管中间出现空洞的情况。不得不感叹好题如好酒越品越深啊！i - stack[-1] - 1) * heights[tmp]这个算法的关键是stack[-1]，而不是pop出来的值，因为中间被pop的值肯定是大于heights[tmp]，所以这个值往前第一个比heights[tmp]小的值才是我们需要的，也就是stack[-1]。

```python
class Solution:
    def largestRectangleArea(self, heights) -> int:
        stack=[]
        heights=[0]+heights+[0]
        res=0
        for i in range(len(heights)):
            while stack and heights[stack[-1]] > heights[i]:
                tmp=stack.pop()
                res=max(res, (i - stack[-1] - 1) * heights[tmp])
            stack.append(i)
        return res
```

[85 最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)[Ye]
题解：https://leetcode-cn.com/problems/maximal-rectangle/solution/python3-qian-zhui-he-dan-diao-zhan-ji-su-vkpp/
这个题实在是太有意思了。可以说找到了一个一般人很难想到的角度来解决问题：就像题解中所说的，我们利用了一个stack来存储一个单调递增的数列，这个数列的作用是存储最低高度，然后关键来了。从后向前计算最低高度条件下可能达到的最大面积。实在没想到单调队列还有这样的作用！
```python
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:return 0
        m,n=len(matrix),len(matrix[0])
        # 记录当前位置上方连续“1”的个数
        pre=[0]*(n+1)
        res=0
        for i in range(m):
            for j in range(n):
                # 前缀和
                pre[j]=pre[j]+1 if matrix[i][j]=="1" else 0

            # 单调栈
            stack=[-1]
            for k,num in enumerate(pre):
                while stack and pre[stack[-1]]>num:
                    index=stack.pop()
                    res=max(res,pre[index]*(k-stack[-1]-1))
                stack.append(k)
        return res
```
[42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)[Li]
```python
#42.接雨水
class Solution:
    def trap(self, height: List[int]) -> int:
        stack,res = [],0
        for i in range(len(height)):
            while stack and height[i]>height[stack[-1]]:
                bottom = stack.pop()
                if not stack: break
                res += (min(height[stack[-1]],height[i])-height[bottom])*(i-stack[-1]-1)
            stack.append(i)
        return res
```
