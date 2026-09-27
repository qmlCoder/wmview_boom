/**
 * 程序的api都绑定在window对象下
 */

/**
 * wmview的核心api
 */
export interface wmapi_cores {
  /**
   * 执行外部程序命令
   */
  run_exe: (source: string, exe: string, args: string[], input: string) => Promise<[boolean, string]>
  /**
   * 显示通知
   */
  notify: (message: string, type: 'success' | 'warning' | 'info' | 'error') => void
  /*添加结果文本 */
  add_reslog: (from: string, text: string) => void
  /**
   * 添加日志文本
   * from: 日志的来源，用于标识
   * text: 日志的内容
   */
  add_logText: (from: string, text: string) => void
  /**
   * 显示加载面板
   */
  show_loading: (tip: string) => void
  /**
   * 隐藏加载面板
   */
  hide_loading: () => void
  /**
   * 保存文本内容到本地文件
   */
  save_text: (path: string, text: string) => void

  read_text: (path: string) => Promise<string>

  save_file_dialog: () => Promise<string | null>

  // 设置底部信息
  set_bottom_message: (message: string) => void

  add_local_plugin: (name: string, code: string, cssPath: string) => void

  // 当前显示的siderl和siderr
  get_show_siders: () => [string, string]

  // 添加markdon文档 path:文档路径 content 内容
  add_appdoc: (path: string, content: string) => void

  // 显示指定路径的文档
  show_doc:(path:string)=>void
}



// src/wmapi.d.ts（基础）
declare global {
  interface Window {
    wmapi_cores: wmapi_cores
  }
}

export interface Info {
  Cub?: CubInfo
  Gjf?: GjfInfo
  Fch?: FChInfo
  Log?: LogInfo
}

export interface CubInfo {
  path: string
  obts: number[]
  size: number[]
  pos0: number[]
  step: number[]
  vmin: number
  vmax: number
  atoms: AtomsProps
}

export interface GjfInfo {
  atoms: AtomsProps
  job: string
  charge: number
  multip: number
}

export interface FChInfo {
  atoms: AtomsProps
}

export interface LogInfo {
  atoms: AtomsProps
  job: string
  engs?: string
}

export interface AtomsProps {
  atoms: string
  atoms_ang: string
}

export interface wmapi_fileinfo {
  get_file_info: (path: string) => Promise<FileInfo>
}

declare global {
  interface Window {
    wmapi_fileinfo: wmapi_fileinfo
  }
}

/**
 * 关于场景的API
 */

// 添加物体时物体的参数类型

// 角度参数
export interface AngleArgs {
  atms:number[]
  acute: boolean
}

// 箭头参数
export interface ArrowArgs {
  pos: [number,number,number]  // pos 箭头起点位置 长度为3的数组
  dir: [number,number,number]  // dir 箭头方向 长度为3的数组
  len: number                  // len 箭头长度
  rad: number                  // rad 箭头半径
  bind: number[]               // bind 箭头绑定的物体索引
  color: string                // color 箭头颜色
}

// 线段参数
export interface LineArgs {
  points: [number,number,number][]  // points 线段的顶点数组
  dashed: boolean     // dashed 是否虚线
}

// 原子轨道参数
export interface PobtArgs {
  position: number[]   // position 添加的原子轨道所在的位置 长度为3的数组
  ratio: number        // ratio 正相位和负相位的大小比例 默认应该为0.5
  direction: number[]  // direction 原子轨道朝向
  scale: number[]      // scale 三个方向的缩放，默认应该为1
}

// 点参数
export interface PointArgs {
  pos: number[]      // pos 点的位置 长度为3的数组
  color: string      // color 点的颜色
  radius: number     // radius 点的半径
}

// 环参数
export interface RingArgs {
  pos: [number, number, number]  // pos 环的位置 长度为3的数组
  dir: [number, number, number]  // dir 环的朝向
  inner_radius: number           // inner_radius 内环半径
  outer_radius: number           // outer_radius 外环半径
  color: string                  // color 环的颜色
}

// 等值面参数
export interface SurfArgs {
  verts: number[]  // verts 顶点数组
  types: number[]  // types 顶点数值类型
  colors: [string,string,string]  // colors 颜色映射数组 负值，零值和正值对应的颜色
  name: string     // name 等值面名称
}

// 局部坐标系参数
export interface SystmArgs {
  cent: [number,number,number]  // cent 坐标系原点位置 长度为3的数组
  // dirs 坐标系三条坐标轴的方向，必须给出 3 个长度为3的单位向量，依次为 x 轴、y 轴、z 轴
  // 例如 [[1,0,0],[0,1,0],[0,0,1]] 表示与全局坐标轴重合
  dirs: [number, number, number][]
  length: number  // length 坐标系长度
}

// 等值面对象（主程序内部对象，这里只给出结构近似，勿依赖其它字段）
export interface Surf {
  uuid: string
}


export interface wmapi_scene {
  /**
   * 设置场景的背景颜色，只能使用hex字符串，例如 #ff0000
   */
  set_color: (color: string) => void
  /**
   * 获取当前显示的分子的名称
   */
  get_mole_name: () => string | undefined

  /**
   * 获取当前分子的结构信息
   * @param mole_name 分子名称
   * @returns 原子信息列表，每个元素为 [原子符号, x, y, z]
   */
  get_mole_geom: (mole_name: string) => [string, number, number, number][]

  // 获取用户选择的原子的索引
  get_atom_select: (mole_name: string) => number[]

  // 设置分子选择的原子 atms:选择的原子
  set_atom_select:(mole_name:string,atms:number[])=>void

  // 添加一个箭头
  add_arrow: (mole_name: string, args: ArrowArgs) => void

  // 添加一条线，可以是实线也可以是虚线
  // #AI 用户可能会让在两个原子之间添加虚线，这个时候使用两个原子的位置作为参数
  add_line: (mole_name: string, args: LineArgs) => void

  // 在指定的分子内添加一个p轨道
  add_pobt: (mole_name: string, args: PobtArgs) => void

  add_surf: (mole_name: string, args: SurfArgs) => string | undefined

  get_surf: (mole_name: string) => Surf | undefined

  /**
   * 在分子指定的位置添加一个局部坐标系
   * @param mole_name 分子名
   * @param args cent: 局部坐标系的中心位置 dirs: 局部坐标系三条坐标轴的方向，依次为 x 轴、y 轴、z 轴，必须给出 3 个长度为3的单位向量
   * @returns
   */
  add_systm: (mole_name: string, args: SystmArgs) => void

  /**
   * 清空一个组
   * @param group_name 组名
   * @returns
   */
  clear_group: (group_name?: string) => void

  on_atom_select:(callback:(mole_name:string,atms:number[])=>void)=>void
}

declare global {
  interface Window {
    wmapi_scene: wmapi_scene
  }
}
// 添加物体时部分物体的默认值



export const ArrowVals: ArrowArgs = {
  pos: [0, 0, 0],
  dir: [0, 0, 1],
  len: 1,
  rad: 0.1,
  bind: [],
  color: '#ff0000',
}

export const PobtVals: PobtArgs = {
  position: [0, 0, 0],
  ratio: 0.5,
  direction: [0, 0, 1],
  scale: [1, 1, 1],
}

export const RingVals: RingArgs = {
  pos: [0, 0, 0],
  dir: [0, 0, 1],
  inner_radius: 0,
  outer_radius: 1,
  color: '#ffffff',
}

export const SystmVals: SystmArgs = {
  cent: [0, 0, 0],
  dirs: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  length: 1
}

/*
分子结构编辑相关的API
原子的索引是从0开始的，用户提示的索引是从1开始的，要转换一下
键长的默认单位是埃
*/

export interface wmapi_editor {
  /**
   * 把键长设置到指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 键长数值
   */
  set_bondLen_to(mole_name: string, atms: [number, number], val: number)

  /**
   * 把键长改变指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 键长改变数值
   */
  set_bondLen_by(mole_name:string,atms:[number,number],val: number)

  /**
   * 把键角设置到指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 键角数值
   */
  set_bondAng_to(mole_name: string, atms: [number, number, number], val: number)

  /**
   * 把键角改变指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 键角改变数值
   */
  set_bondAng_by(mole_name: string, atms: [number, number, number], val: number)

  /**
   * 把面角设置到指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 面角数值
   */
  set_faceAng_to(mole_name: string, atms: [number, number, number, number], val: number)

  /**
   * 把面角改变指定值
   * @param mole_name 分子名
   * @param atms 原子列表
   * @param val 面角改变数值
   */
  set_faceAng_by(mole_name: string, atms: [number, number, number, number], val: number)

  /**
   * 保存为GJF文件
   * @param mole_name 分子名
   */
  save_gjf(mole_name: string)

  /**
   * 设置原子位置
   * @param mole_name 分子名
   * @param atom_position 原子位置记录
   */
  set_atom_position(mole_name:string,atom_position:Record<number, [number, number, number]>)
}

declare global {
  interface Window {
    wmapi_editor: wmapi_editor
  }
}

export interface wmapi_script {
  /**
   * 执行脚本
   * @param script 需要执行的脚本的字符串
   * @returns 执行是否成功，成功返回true，失败返回false；
   * @returns 执行结果的字符串
   */
  run_script: (script: string) => Promise<[boolean, string]>

  /**
   * 获取API文档
   * @param plugin 插件名称，若不指定则获取全部
   * @returns API文档的字符串
   */
  get_apidoc: (plugin?: string) => string
}

declare global {
  interface Window {
    wmapi_script: wmapi_script
  }
}

/*
波函数分析功能的API
函数会返回计算是否成功以及失败成功的结果内容，如果成功的话尝试分析一下返回的结果
*/

type FukuiType = 'f-' | 'f+' | 'f0' | 'df'
type ChargeType = 'mulliken' | 'lowdin' | 'hirshfeld'
type OrderType = 'mayer' | 'wiberg' // 键级的类型
export type GridFuncType =
  | '分子轨道'
  | '分子电子密度'
  | '轨道电子密度'
  | '自由电子密度' // 也可以叫预电子密度
  | 'π电子密度_pocv'
  | 'π电子密度_mocv'
  | 'RDG'
  | 'IRI'
  | '电子拉普拉斯'
  | 'sign(λ2)ρ'
  | '变形电子密度'
  | '福井函数'
  | '拉格朗日动能密度'
  | '哈密顿动能密度'
  | '动能密度'
  | 'ELF'
  | 'LOL'
  | '电子信息熵'
  | '活性电子'
  | '电子曲率'
  | '电子密度差值'

export interface wmapi_wfana {
  atomprop: {
    /**
     * 计算指定名称分子的基础电子布居
     * 如果当前场景显示的不是这个分子的话，则要先切换到这个分子
     * @param mole_name 分子名称
     * @param ctype 计算电子电荷的类型
     * @returns 函数执行是否成功，成功返回true，失败返回false；成功或失败返回的结果
     */
    basic_charge: (mole_name: string, ctype: ChargeType) => Promise<[boolean, string]>

    /**
     * 计算分子的pi电子布居
     * @param mole_name 分子名
     * @param method 计算方法
     * @param ctype 计算电子电荷的类型
     * @returns 函数执行是否成功，成功返回true，失败返回false；成功或失败返回的结果
     */
    pi_electron_polulation: (mole_name: string, method: 'pocv' | 'mocv', ctype: ChargeType) => Promise<[boolean, string]>

    /**
     * 计算指定分子的福井函数，总共需要三个分子
     * #AI 如果用户没有指定其它两个分子，则根据文件列表中分子名猜测，如果猜不出来则让用户指定
     * @param mole_name 要计算的分子的名称，一般为中性分子
     * @param mole_name_n 多一个电子的分子 n:negative 可以为中性分子名加后缀 _N 或者后缀 _N+1(多一个电子)
     * @param mole_name_p 多一个电子的分子 p:positive 可以为中性分子名加后缀 _P 或者后缀 _N-1(少一个电子)
     * @param fukui_type 福井函数类型
     * @param ctype 电荷类型，使用哪种方式计算原子电荷
     * @returns
     */
    fukui_function: (mole_name: string, mole_name_n: string, mole_name_p: string, fukui_type: FukuiType, ctype: ChargeType) => Promise<[boolean, string]>
  }
  bondprop: {
    /*
     * 计算指定分子的键级，所有的键会同时计算
     * @param mole_name 分子名
     * @param order_type 键级类型
     */
    basis_bond_order: (mole_name: string, order_type: OrderType) => Promise<[boolean, string]>

    /**
     * 计算分子的轨道键级，用于衡量某个分子轨道内的两个原子之间是成键(正值)还是反键(负值)
     * @param mole_name 分子名
     * @param order_type 键级类型
     * @param iobt 轨道索引
     * @returns
     */
    orbital_bond_order: (mole_name: string, order_type: OrderType, iobt: number) => Promise<[boolean, string]>

    /**
     * 计算分子pi键级
     * @param mole_name 分子名
     * @param order_type 键级类型
     * @param method 计算方法
     * @returns
     */
    pi_bond_order: (mole_name: string, order_type: OrderType, method: 'pocv' | 'mocv') => Promise<[boolean, string]>

    /**
     * 计算两原子之间的换轴键级
     * @param mole_name 分子名
     * @param order_type 键级类型
     * @param method 计算方法
     * @param atms 原子列表
     * @param is_pi 是否为pi键
     * @returns
     * #AI 当分子选择的有原子的时候使用选择的原子(wmapi_scene)，若未指定则需要用户指定
     */
    circle_bond_order: (mole_name: string, order_type: OrderType, method: 'pocv' | 'mocv', atms: [number, number],is_pi:boolean) => Promise<[boolean, string]>
  }
  gridprop: {
    // 显示分子轨道，iobt为轨道索引
    show_mole_orbital: (mole_name: string,iobt:number) => Promise<[boolean, string]>
    // 构建一个函数的等值面，如果成功返回等值面的uuid
    build_iso_surf: (mole_name: string, func_name: GridFuncType) => Promise<[boolean, string]>
    /**
     * 用另一个函数给等值面上色
     * @param dest 着色目标，'surf'为等值面，'plane'为平面
     * @param mole_name 分子名
     * @param func_name 函数名
     * @param uuid 如果不传入等值面uuid则使用最后一个等值面
     * @returns
     */
    color_iso_surf: (dest: 'surf' | 'plane', mole_name: string, func_name: GridFuncType, uuid?: string) => Promise<[boolean, string]>
    // 获取每个分子轨道的pi电子数，可以用来衡量哪些分子轨道最像pi分子轨道
    get_obt_pi_popul: () => Promise<number[]|undefined>
  }
}

declare global {
  interface Window {
    wmapi_wfana: wmapi_wfana
  }
}



/**
 * 关于文件的API
 */

 export type MoleInfo = {
   syms: string[]
   xyzs: [number,number,number][]
   bonds: [number,number][]
 }


export interface wmapi_files {
  /**
   * 获取程序的根文件夹
   */
  get_rootFold: () => string
  /**
   * 等待程序根目录初始化完成(异步获取),返回根目录。读取 root/plugs 相关路径前应先调用
   */
  wait_root: () => Promise<string>
  /**
   * 获取分子文件夹
   */
  get_moleFold: () => string
  /**
   * 获取插件文件夹
   */
  get_plugFold: () => string

  /**
   * 获取当前分子文件夹下的所有分子名称
   * @returns 分子名称列表
   */
  get_file_list: () => string[]

  /**
   * 获取指定分子的信息
   * @param name 分子文件名
   * @returns 分子信息
   */
  get_mole_info: (name: string) => Promise<MoleInfo>

  /**
   * 显示指定名称的分子文件
   * @param name 分子文件名
   * @param show 是否显示
   */
  load_file: (name: string,show:boolean) => void

  // get_moleInfo: (name: string) => MoleInfo

  // 获取选择的文件 vue Ref<string>
  get_select: () => string

  // 获取多选的文件 vue Ref<string[]>
  get_option: () => string[]

  // 选择文件变化时触发，获取先择的文件名
  on_select: (callback: (select: string) => void) => void

  // 多选文件变化时触发，获取先择的文件名
  on_option:(callback:(option: string[]) => void) => void
}

declare global {
  interface Window {
    wmapi_files: wmapi_files
  }
}
