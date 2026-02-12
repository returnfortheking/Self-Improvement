#!/usr/bin/env python3
"""验证 DETAILED_self_and_methods.md 中的关键结论"""

print("=" * 60)
print("测试 1: 实例方法访问类属性")
print("=" * 60)

class Demo:
    class_var = "类变量"

    def method_via_class_name(self):
        return Demo.class_var

    def method_via_self_class(self):
        return self.__class__.class_var

    def method_via_self(self):
        """文档中遗漏的最常见方式"""
        return self.class_var

d = Demo()

print("方式1 - 通过类名:", d.method_via_class_name())
print("方式2 - 通过self.__class__:", d.method_via_self_class())
print("方式3 - 通过self:", d.method_via_self())

print("\n" + "=" * 60)
print("测试 2: 修改类属性的陷阱")
print("=" * 60)

class Test:
    class_var = "原始值"

    def modify_via_self(self):
        """这样会创建实例属性，不是修改类属性"""
        self.class_var = "修改后的值"

    def modify_via_class_name(self):
        """这才是修改类属性"""
        Test.class_var = "真正的修改"

t1 = Test()
t2 = Test()

print(f"修改前 t1.class_var: {t1.class_var}")
print(f"修改前 t2.class_var: {t2.class_var}")
print(f"修改前 Test.class_var: {Test.class_var}")

t1.modify_via_self()

print(f"\nt1.modify_via_self() 后:")
print(f"t1.class_var: {t1.class_var}")  # 修改后的值（实例属性）
print(f"t2.class_var: {t2.class_var}")  # 原始值（类属性未变）
print(f"Test.class_var: {Test.class_var}")  # 原始值（类属性未变）

t1.modify_via_class_name()

print(f"\nt1.modify_via_class_name() 后:")
print(f"t1.class_var: {t1.class_var}")  # 还是实例属性
print(f"t2.class_var: {t2.class_var}")  # 真正的修改（类属性）
print(f"Test.class_var: {Test.class_var}")  # 真正的修改（类属性）

print("\n" + "=" * 60)
print("结论：")
print("1. ✅ 实例方法可以访问类属性（通过 self.类属性名）")
print("2. ⚠️ 实例方法修改类属性有陷阱，应该用类名或 self.__class__")
print("3. ❌ 文档中遗漏了 'self.类属性名' 这种最常见的方式")
print("=" * 60)
