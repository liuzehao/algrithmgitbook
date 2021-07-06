# 三.滑动窗口
和回溯算法相似，滑动窗口也是一种比较成熟的模版。
## [例一3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)
```python
def sttr(strr):
    windows=dict()
    right=0
    left=0
    res=0
    while right<len(strr):
        c=strr[right]
        right+=1
        if c in windows:
            windows[c]=windows[c]+1
        else:
            windows[c]=1
        while windows[c]>1:
            d=strr[left]
            left+=1
            windows[d]=windows[d]-1
        res=max(res,right-left)
    return res
        
print(sttr("pwwwwwwwkkew"))
```

## [例二.最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)
```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        windows=dict()
        needs=dict()
        for i in t:
            if i in needs:
                needs[i]=needs[i]+1
            else:
                needs[i]=1
        right=left=0
        void=0
        res=float("inf")
        start=0
        while right<len(s):
            c=s[right]
            right+=1
            if c in needs:
                if c in windows:
                    windows[c]=windows[c]+1
                else:
                    windows[c]=1
                if windows[c]==needs[c]:
                    void+=1
                while void==len(needs):
                    d=s[left]
                    left+=1
                    if right-left<res:
                        start=left
                        res=right-left
                    if d in needs:
                        if needs[d]==windows[d]:
                            void-=1
                        windows[d]=windows[d]-1
        return s[start-1:start+res] if res!=float("inf") else ""
```

## [例三438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

```python
def strfind(s,p):
    windows=dict()
    needs=dict()
    for i in p:
        if i in needs:
            needs[i]=needs[i]+1
        else:
            needs[i]=1
    right=left=void=0
    res=[]
    while right<=len(s)-1:
        c=s[right]
        right+=1
        if c in needs:
            if c in windows:
                windows[c]=windows[c]+1
            else:
                windows[c]=1
            if windows[c]==needs[c]:
                void+=1
            while right-left>=len(p):
                if void==len(needs) and right-left==len(p):
                    res.append(left)
                d=s[left]
                left+=1
                if d in needs:
                    if windows[d]== needs[d]:
                        void-=1
                    windows[d]=windows[d]-1
    return res

print(strfind("abccecdt","ce"))
```
对于子串，固定滑窗的大小遍历复杂度更加低。
```python
#References to DaLi 2021.1.19 
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        res = []
        n = len(p)
        d1 = dict(Counter(p))
        d2 = dict(Counter(s[:n]))
        for i in d2:
            if i in d1:
                d1[i]-=d2[i]
        for j in range(len(s)-n+1):
            if max(d1.values())==0:
                res.append(j)
            if j==len(s)-n: #右窗口到达边界，不再进行下面操作
                continue
            if s[j] in d1:
                d1[s[j]]+=1
            if s[j+n] in d1:
                d1[s[j+n]]-=1
        return res

```
## [例四.字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        windows=dict()
        needs=dict()
        for i in s1:
            if i in needs:
                needs[i]=needs[i]+1
            else:
                needs[i]=1
        right=left=0
        voiad=0
        while right<len(s2):
            c=s2[right]
            right+=1
            if c in needs:
                if c in windows:
                    windows[c]=windows[c]+1
                else:
                    windows[c]=1
                if windows[c]==needs[c]:
                    voiad+=1
                while voiad==len(needs):
                    d=s2[left]
                    if right-left==len(s1):
                        return True
                    if d in needs:
                        if windows[d]==needs[d]:
                            voiad-=1
                        windows[d]=windows[d]-1
                    left=left+1
        return False
```

## 题整合
[567. 字符串的排列](https://leetcode-cn.com/problems/permutation-in-string/)