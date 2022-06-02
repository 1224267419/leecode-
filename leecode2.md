#### [剑指 Offer II 035. 最小时间差](https://leetcode.cn/problems/569nqc/)

先排序，再求时间

```python

def getMinutes(t: str) -> int:
    return ((ord(t[0]) - ord('0')) * 10 + ord(t[1]) - ord('0')) * 60 + (ord(t[3]) - ord('0')) * 10 + ord(t[4]) - ord('0')

class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        n = len(timePoints)
        if n > 1440:
            return 0
        timePoints.sort()
        ans = float('inf')
        t0Minutes = getMinutes(timePoints[0])
        preMinutes = t0Minutes
        for i in range(1, n):
            minutes = getMinutes(timePoints[i])
            ans = min(ans, minutes - preMinutes)  # 相邻时间的时间差
            preMinutes = minutes
        ans = min(ans, t0Minutes + 1440 - preMinutes)  # 首尾时间的时间差
        return ans

```





#### [剑指 Offer II 036. 后缀表达式](https://leetcode.cn/problems/8Zf90G/) [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/) 字符串,不会

用栈的先进后出来表达波兰后缀表达式,**重点是要理解这玩意得用栈(先进后出)**

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        op_to_binary_fn = {
            "+": add,
            "-": sub,
            "*": mul,
            "/": lambda x, y: int(x / y),   # 需要注意 python 中负数除法的表现与题目不一致
        }

        n = len(tokens)
        stack = [0] * ((n + 1) // 2)
        index = -1
        for token in tokens:
            try:
                num = int(token)
                index += 1
                stack[index] = num
            except ValueError:
                index -= 1
                stack[index] = op_to_binary_fn[token](stack[index], stack[index + 1])
            
        return stack[0]
```

#### [4. 寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/) 不会时间O(log(m+n))

有log必然是二分法得到答案，但具体是怎么实现的呢

