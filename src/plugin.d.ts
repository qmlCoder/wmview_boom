import type { Component } from 'vue'
const DeviceType='android'|'windows'

export interface Sider { // 侧边栏
  name: string // 名称
  icon: string // 图标
  comp: Component // 组件
}
export interface Setting { // 设置界面（会被放在程序的设置界面中）
  name: string // 名称
  comp: Component // 组件
}
export interface Plugin {
  name: string // 插件名称
  sider: Sider // 侧边栏
  setting?: Setting // 设置界面
  scene_color?: string // 场景颜色
  apidoc?: string // 场景的API文档
  position: 'left' | 'right' // 位置
  kind?: 'builtin' | 'local' | 'dev' // 类型
  device: DeviceType[]
}
