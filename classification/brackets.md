# 括号问题
括号问题是一个很有趣的系列问题，其中三道问题：
括号生成是一个回溯问题
有效的括号是一个stack问题
最长括号有效括号是一个dp问题
都很经典。
除此之外还有：删除无效的括号。

[22.括号生成](https://leetcode-cn.com/problems/generate-parentheses/)
回溯；动规
```python
#22.括号生成
class Solution:
    def generateParenthesis(self, n: int):
        #回溯法
        '''res = []
        def func(temp,l,r):
            if len(temp)==2*n:
                res.append(''.join(temp))
                return
            if l<n:
                func(temp+['('],l+1,r)
            if r<l:
                func(temp+[')'],l,r+1)
        func([],0,0)
        return res'''
        
        #动规解法
        #从dp[i-1]转移到dp[i],新增一个括号，括住了m个括号,有k=i-1-m个括号放右边
        #dp[i]="("+dp[m]+")"+dp[k]
        #其中m+k=i-1
        if n == 0:
            return []
        dp = [[''],["()"]]
        for i in range(2,n+1):    # 开始计算i组括号时的括号组合
            temp = []        
            for m in range(i):    # 开始从0到n-1遍历m ，其中m+k=i-1 作为索引
                list1 = dp[m]        # m个括号时组合情况
                list2 = dp[i-1-m]    # k = (i-1) - m 时的括号组合情况
                for k1 in list1:  
                    for k2 in list2:
                        el = "(" + k1 + ")" + k2
                        temp.append(el)    # 把所有可能的情况添加到 temp 中
            dp.append(temp)    # 这个temp就是i组括号的所有情况
        return dp[n]
```

[有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)
栈
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
本题要注意添加‘？’是用来防止程序报错的。stack.pop意味着stack不能是空的，如果是空的会报错。dit[]意味着dic中必须有'?'这个key，不然也会报错。整个程序的逻辑是只要字符串s中出现了非括号元素一定报错；当出现左括号就入栈，出现对应右括号就出栈。最后考虑边界问题:"?"是不是会导致出现bug的情况出现，事实上并不会，这是一个巧妙的设计。由于"?"的key和vaue相同，并不能进入出栈的代码，因此没有导致bug。

[32. 最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)
动规
```python
#32. 最长有效括号
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = [0]*len(s)
        res = 0
        for i in range(1,len(s)):
            #加在右边
            if s[i]==')' and s[i-1]=='(':
                dp[i] = dp[i-2]+2
            #括到中间
            if s[i]==s[i-1]==')' and s[i-dp[i-1]-1]=='(' and i>dp[i-1] : #i>dp[i-1]处理边界
                dp[i] = dp[i-1]+dp[i-dp[i-1]-2]+2
            if dp[i]>res:
                res=dp[i]
        return res
```
[301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)
