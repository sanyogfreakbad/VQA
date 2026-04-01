import { useState, useRef, useCallback } from 'react';

interface WebPosition {
  x: number
  y: number
  width: number
  height: number
}

interface ComparisonItem {
  serial_number?: number
  element: string
  text: string
  sub_type: string
  figma_value: string | number
  web_value: string | number
  delta: string
  severity: string
  web_position: WebPosition
  web_node_id: string
  web_locator: string
}

interface ByCategory {
  [key: string]: ComparisonItem[]
}

interface CategorySummary {
  text: number
  spacing: number
  size: number
  missing_elements: number
  other: number
}

interface ComparisonResult {
  by_category: ByCategory
  summary?: {
    total_differences: number
    categories?: CategorySummary
  }
  web_screenshot_url?: string
  figma_screenshot_url?: string
}

interface FormData {
  figmaUrl: string
  webUrl: string
  loginUrl: string
  username: string
  password: string
  useGemini: boolean
}

interface Annotation {
  x: number
  y: number
  width: number
  height: number
  category: string
  element: string
  locator: string
  node_id: string
  issues: string[]
  serial_numbers: number[]
}

interface CategoryColors {
  [key: string]: {
    border: string
    bg: string
    text: string
  }
}

interface AnnotatedImageState {
  baseImageUrl: string | null
  annotations: Annotation[]
  categoryColors: CategoryColors
  loading: boolean
  error: string | null
  imageWidth: number
  imageHeight: number
}

const CATEGORY_CONFIG: Record<string, { label: string; badgeClass: string; rowBorder: string; tagBg: string; tagText: string }> = {
  text: {
    label: 'Text',
    badgeClass: 'bg-violet-100 text-violet-800 border border-violet-300',
    rowBorder: 'border-l-[3px] border-l-violet-500',
    tagBg: 'bg-violet-100',
    tagText: 'text-violet-800',
  },
  spacing: {
    label: 'Spacing',
    badgeClass: 'bg-pink-100 text-pink-800 border border-pink-300',
    rowBorder: 'border-l-[3px] border-l-pink-500',
    tagBg: 'bg-pink-100',
    tagText: 'text-pink-800',
  },
  size: {
    label: 'Size',
    badgeClass: 'bg-cyan-100 text-cyan-800 border border-cyan-300',
    rowBorder: 'border-l-[3px] border-l-cyan-500',
    tagBg: 'bg-cyan-100',
    tagText: 'text-cyan-800',
  },
  missing_elements: {
    label: 'Missing',
    badgeClass: 'bg-red-100 text-red-700 border border-red-300',
    rowBorder: 'border-l-[3px] border-l-red-500',
    tagBg: 'bg-red-100',
    tagText: 'text-red-700',
  },
  padding: {
    label: 'Padding',
    badgeClass: 'bg-amber-100 text-amber-700 border border-amber-300',
    rowBorder: 'border-l-[3px] border-l-amber-500',
    tagBg: 'bg-amber-100',
    tagText: 'text-amber-700',
  },
  color: {
    label: 'Color',
    badgeClass: 'bg-emerald-100 text-emerald-800 border border-emerald-300',
    rowBorder: 'border-l-[3px] border-l-emerald-500',
    tagBg: 'bg-emerald-100',
    tagText: 'text-emerald-800',
  },
  components: {
    label: 'Components',
    badgeClass: 'bg-orange-100 text-orange-800 border border-orange-300',
    rowBorder: 'border-l-[3px] border-l-orange-500',
    tagBg: 'bg-orange-100',
    tagText: 'text-orange-800',
  },
  buttons_cta: {
    label: 'Buttons',
    badgeClass: 'bg-sky-100 text-sky-800 border border-sky-300',
    rowBorder: 'border-l-[3px] border-l-sky-500',
    tagBg: 'bg-sky-100',
    tagText: 'text-sky-800',
  },
  other: {
    label: 'Other',
    badgeClass: 'bg-amber-100 text-amber-700 border border-amber-300',
    rowBorder: 'border-l-[3px] border-l-amber-500',
    tagBg: 'bg-amber-100',
    tagText: 'text-amber-700',
  },
}

const OTHER_CATEGORIES = ['padding', 'color', 'components', 'buttons_cta']

function App() {
  const [formData, setFormData] = useState<FormData>({
    figmaUrl: '',
    webUrl: '',
    loginUrl: '',
    username: '',
    password: '',
    useGemini: false
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ComparisonResult | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedSerialNumbers, setSelectedSerialNumbers] = useState<Set<number>>(new Set())
  const [hoveredSerial, setHoveredSerial] = useState<number | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<AnnotatedImageState>({
    baseImageUrl: null,
    annotations: [],
    categoryColors: {},
    loading: false,
    error: null,
    imageWidth: 0,
    imageHeight: 0
  })
  const [figmaImageUrl, setFigmaImageUrl] = useState<string | null>(null)
  const [splitView, setSplitView] = useState(false)
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }))
  }

  const handleCompare = async () => {
    if (!formData.figmaUrl || !formData.webUrl) {
      setError('Figma URL and Web URL are required')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setSelectedCategory(null)
    setSelectedSerialNumbers(new Set())
    setHoveredSerial(null)
    setSplitView(false)
    setAnnotatedImage({
      baseImageUrl: null,
      annotations: [],
      categoryColors: {},
      loading: false,
      error: null,
      imageWidth: 0,
      imageHeight: 0
    })
    setFigmaImageUrl(null)

    const requestBody = {
      figma_url: formData.figmaUrl,
      web_url: formData.webUrl,
      login_url: formData.loginUrl || undefined,
      credentials: formData.username ? {
        username: formData.username,
        password: formData.password,
        selectors: {
          submit: "[role='button']:has-text('Sign In')"
        }
      } : undefined,
      post_login_steps: [
        { action: "wait", duration: 2000 },
        { action: "click", selector: ".css-ai6why-control", nth: 0 },
        { action: "click", text: "DPW CIC CB Enterprises" },
        { action: "click", selector: ".css-ai6why-control", nth: 0 },
        { action: "click", text: "CIC - CB Warehouse 1" },
        { action: "click", test_id: "next" }
      ],
      wait_for_selector: "body",
      viewport: { width: 1440, height: 800 },
      max_depth: 50,
      use_gemini: formData.useGemini
    }

    try {
      const response = await fetch('/api/compare/urls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleShowImage = async () => {
    if (!result) return

    setAnnotatedImage(prev => ({ ...prev, loading: true, error: null }))

    try {
      const metadataResponse = await fetch('/api/annotate/metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comparison_results: result })
      })

      if (!metadataResponse.ok) {
        const errorData = await metadataResponse.json().catch(() => ({}))
        throw new Error(errorData.detail || `HTTP error! status: ${metadataResponse.status}`)
      }

      const metadataData = await metadataResponse.json()

      let baseImageUrl = result.web_screenshot_url

      if (!baseImageUrl) {
        const annotateResponse = await fetch('/api/annotate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            comparison_results: result,
            web_url: formData.webUrl,
            login_url: formData.loginUrl || undefined,
            credentials: formData.username ? {
              username: formData.username,
              password: formData.password,
              selectors: { submit: "[role='button']:has-text('Sign In')" }
            } : undefined,
            post_login_steps: [
              { action: "wait", duration: 2000 },
              { action: "click", selector: ".css-ai6why-control", nth: 0 },
              { action: "click", text: "DPW CIC CB Enterprises" },
              { action: "click", selector: ".css-ai6why-control", nth: 0 },
              { action: "click", text: "CIC - CB Warehouse 1" },
              { action: "click", test_id: "next" }
            ],
            viewport: { width: 1440, height: 800 },
            include_metadata: true
          })
        })

        if (!annotateResponse.ok) {
          const errorData = await annotateResponse.json().catch(() => ({}))
          throw new Error(errorData.detail || `HTTP error! status: ${annotateResponse.status}`)
        }

        const annotateData = await annotateResponse.json()
        baseImageUrl = annotateData.annotated_screenshot_url
      }

      setAnnotatedImage({
        baseImageUrl: baseImageUrl || null,
        annotations: metadataData.annotations || [],
        categoryColors: metadataData.category_colors || {},
        loading: false,
        error: null,
        imageWidth: 0,
        imageHeight: 0
      })

      setFigmaImageUrl(result.figma_screenshot_url || null)
    } catch (err) {
      setAnnotatedImage(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load annotated image'
      }))
    }
  }

  const handleImageLoad = useCallback(() => {
    if (imageRef.current) {
      setAnnotatedImage(prev => ({
        ...prev,
        imageWidth: imageRef.current?.naturalWidth || 0,
        imageHeight: imageRef.current?.naturalHeight || 0
      }))
    }
  }, [])

  const toggleSerialNumber = (serial: number) => {
    setSelectedSerialNumbers(prev => {
      const newSet = new Set(prev)
      if (newSet.has(serial)) {
        newSet.delete(serial)
      } else {
        newSet.add(serial)
      }
      return newSet
    })
  }

  const clearSerialSelection = () => {
    setSelectedSerialNumbers(new Set())
  }

  const getFilteredAnnotations = useCallback((): Annotation[] => {
    let filtered = annotatedImage.annotations

    if (selectedCategory && selectedCategory !== 'total') {
      if (selectedCategory === 'other') {
        filtered = filtered.filter(ann =>
          OTHER_CATEGORIES.includes(ann.category) ||
          !['text', 'spacing', 'size', 'missing_elements'].includes(ann.category)
        )
      } else {
        filtered = filtered.filter(ann => ann.category === selectedCategory)
      }
    }

    if (selectedSerialNumbers.size > 0) {
      filtered = filtered.filter(ann =>
        ann.serial_numbers.some(sn => selectedSerialNumbers.has(sn))
      )
    }

    return filtered
  }, [annotatedImage.annotations, selectedCategory, selectedSerialNumbers])

  const getAnnotationColor = (category: string): string => {
    const colors = annotatedImage.categoryColors[category]
    return colors?.border || '#f59e0b'
  }

  const isAnnotationHighlighted = (ann: Annotation): boolean => {
    if (hoveredSerial === null) return false
    return ann.serial_numbers.includes(hoveredSerial)
  }

  const getAllItems = (): { category: string; item: ComparisonItem }[] => {
    if (!result?.by_category) return []

    const items: { category: string; item: ComparisonItem }[] = []
    Object.entries(result.by_category).forEach(([category, categoryItems]) => {
      categoryItems.forEach(item => {
        items.push({ category, item })
      })
    })
    return items
  }

  const getFilteredItems = (): { category: string; item: ComparisonItem }[] => {
    const allItems = getAllItems()
    if (!selectedCategory) return allItems

    if (selectedCategory === 'total') return allItems

    if (selectedCategory === 'other') {
      return allItems.filter(({ category }) => {
        const normalizedCategory = category.toLowerCase().replace(/\s+/g, '_')
        return OTHER_CATEGORIES.includes(normalizedCategory) ||
          !['text', 'spacing', 'size', 'missing_elements'].includes(normalizedCategory)
      })
    }

    return allItems.filter(({ category }) => {
      const normalizedCategory = category.toLowerCase().replace(/\s+/g, '_')
      return normalizedCategory === selectedCategory
    })
  }

  const handleCategoryClick = (category: string) => {
    setSelectedCategory(prev => prev === category ? null : category)
  }

  const getCategoryCounts = (): Record<string, number> => {
    if (!result?.by_category) return {}

    const counts: Record<string, number> = {
      text: 0,
      spacing: 0,
      size: 0,
      missing_elements: 0,
      other: 0
    }

    Object.entries(result.by_category).forEach(([category, items]) => {
      const normalizedCategory = category.toLowerCase().replace(/\s+/g, '_')
      if (Object.prototype.hasOwnProperty.call(counts, normalizedCategory)) {
        counts[normalizedCategory] = items.length
      } else if (OTHER_CATEGORIES.includes(normalizedCategory)) {
        counts.other += items.length
      } else {
        counts.other += items.length
      }
    })

    return counts
  }

  const getTotalDifferences = (): number => {
    if (!result?.by_category) return 0
    return Object.values(result.by_category).reduce((sum, items) => sum + items.length, 0)
  }

  const getCategoryConfig = (category: string) => {
    const normalizedCategory = category.toLowerCase().replace(/\s+/g, '_')
    return CATEGORY_CONFIG[normalizedCategory] || CATEGORY_CONFIG.other
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans flex flex-col">
      {/* Header */}
      <header className="bg-white px-4 md:px-8 py-4 border-b border-gray-200 shadow-sm">
        <div className="max-w-[1600px] mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl md:text-2xl font-semibold text-slate-800">Figma vs Web Comparator</h1>
            <p className="text-gray-500 text-sm">Compare your Figma designs with live web implementations</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col max-w-[1600px] w-full mx-auto p-4 md:p-6 overflow-y-auto">
        {/* Form Section */}
        <section className="shrink-0 mb-6">
          <div className="bg-white rounded-lg p-4 md:p-6 border border-gray-200 shadow-sm">
            <h2 className="text-base font-semibold mb-4 text-slate-800 pb-3 border-b border-gray-200">
              Configuration
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label htmlFor="figmaUrl" className="block mb-1.5 font-medium text-gray-700 text-[0.8125rem]">
                  Figma URL *
                </label>
                <input
                  type="url"
                  id="figmaUrl"
                  name="figmaUrl"
                  value={formData.figmaUrl}
                  onChange={handleInputChange}
                  placeholder="https://www.figma.com/design/..."
                  className="w-full px-3.5 py-2.5 rounded-md border border-gray-300 bg-white text-gray-800 text-sm transition-all focus:outline-none focus:border-blue-600 focus:ring-[3px] focus:ring-blue-600/10 placeholder:text-gray-400"
                />
              </div>

              <div>
                <label htmlFor="webUrl" className="block mb-1.5 font-medium text-gray-700 text-[0.8125rem]">
                  Web URL *
                </label>
                <input
                  type="url"
                  id="webUrl"
                  name="webUrl"
                  value={formData.webUrl}
                  onChange={handleInputChange}
                  placeholder="https://example.com/page"
                  className="w-full px-3.5 py-2.5 rounded-md border border-gray-300 bg-white text-gray-800 text-sm transition-all focus:outline-none focus:border-blue-600 focus:ring-[3px] focus:ring-blue-600/10 placeholder:text-gray-400"
                />
              </div>

              <div>
                <label htmlFor="loginUrl" className="block mb-1.5 font-medium text-gray-700 text-[0.8125rem]">
                  Login URL
                </label>
                <input
                  type="url"
                  id="loginUrl"
                  name="loginUrl"
                  value={formData.loginUrl}
                  onChange={handleInputChange}
                  placeholder="https://example.com/login"
                  className="w-full px-3.5 py-2.5 rounded-md border border-gray-300 bg-white text-gray-800 text-sm transition-all focus:outline-none focus:border-blue-600 focus:ring-[3px] focus:ring-blue-600/10 placeholder:text-gray-400"
                />
              </div>
            </div>

            {/* Credentials */}
            <div className="mt-4 p-4 bg-gray-50 rounded-md border border-gray-200">
              <h3 className="text-[0.8125rem] font-semibold mb-3 text-gray-700">Credentials (optional)</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="username" className="block mb-1.5 font-medium text-gray-700 text-[0.8125rem]">
                    Username
                  </label>
                  <input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    placeholder="Enter username"
                    className="w-full px-3.5 py-2.5 rounded-md border border-gray-300 bg-white text-gray-800 text-sm transition-all focus:outline-none focus:border-blue-600 focus:ring-[3px] focus:ring-blue-600/10 placeholder:text-gray-400"
                  />
                </div>
                <div>
                  <label htmlFor="password" className="block mb-1.5 font-medium text-gray-700 text-[0.8125rem]">
                    Password
                  </label>
                  <input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    placeholder="Enter password"
                    className="w-full px-3.5 py-2.5 rounded-md border border-gray-300 bg-white text-gray-800 text-sm transition-all focus:outline-none focus:border-blue-600 focus:ring-[3px] focus:ring-blue-600/10 placeholder:text-gray-400"
                  />
                </div>
              </div>
            </div>

            {/* Options */}
            <div className="mt-4 p-4 bg-gray-50 rounded-md border border-gray-200">
              <h3 className="text-[0.8125rem] font-semibold mb-3 text-gray-700">Options</h3>
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="useGemini"
                  name="useGemini"
                  checked={formData.useGemini}
                  onChange={handleInputChange}
                  className="w-[18px] h-[18px] mt-0.5 cursor-pointer accent-blue-600"
                />
                <label htmlFor="useGemini" className="flex flex-col gap-1 font-medium text-gray-700 text-sm cursor-pointer">
                  Use Gemini AI Refinement
                  <span className="font-normal text-xs text-gray-500">
                    Enable AI-powered visual validation to reduce false positives
                  </span>
                </label>
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end mt-4 pt-4 border-t border-gray-200 gap-3">
              <button
                onClick={handleCompare}
                disabled={loading}
                className="px-6 py-2.5 rounded-md border-none bg-blue-600 text-white text-sm font-medium cursor-pointer transition-all inline-flex items-center justify-center gap-2 hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 rounded-full border-t-white animate-spin" />
                    Comparing...
                  </>
                ) : (
                  'Compare'
                )}
              </button>
              {result && (
                <button
                  onClick={handleShowImage}
                  disabled={annotatedImage.loading}
                  className="px-6 py-2.5 rounded-md border-2 border-violet-500 bg-white text-violet-500 text-sm font-medium cursor-pointer transition-all inline-flex items-center justify-center gap-2 hover:bg-violet-500 hover:text-white disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {annotatedImage.loading ? (
                    <>
                      <span className="w-4 h-4 border-2 border-violet-500/30 rounded-full border-t-violet-500 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    'Show Image'
                  )}
                </button>
              )}
            </div>

            {error && (
              <div className="mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-md text-red-600 text-sm">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Annotated Image Section */}
        {(annotatedImage.baseImageUrl || annotatedImage.loading || annotatedImage.error) && (
          <section className="animate-fadeIn mb-6">
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-4 md:p-6">
              <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                <h2 className="text-base font-semibold text-slate-800">Annotated Screenshot</h2>
                <div className="flex items-center gap-4 flex-wrap">
                  {figmaImageUrl && (
                    <button
                      onClick={() => setSplitView(!splitView)}
                      className={`px-4 py-2 rounded-md border-2 border-indigo-500 text-[0.8125rem] font-semibold cursor-pointer transition-all inline-flex items-center gap-1.5 ${
                        splitView
                          ? 'bg-indigo-500 text-white hover:bg-indigo-600'
                          : 'bg-white text-indigo-500 hover:bg-indigo-50'
                      }`}
                    >
                      {splitView ? 'Single View' : 'Split View'}
                    </button>
                  )}
                  {annotatedImage.annotations.length > 0 && (
                    <div className="flex items-center gap-4">
                      <span className="text-[0.8125rem] text-gray-500">
                        Showing {getFilteredAnnotations().length} of {annotatedImage.annotations.length} boxes
                      </span>
                      {selectedSerialNumbers.size > 0 && (
                        <button
                          onClick={clearSerialSelection}
                          className="px-3 py-1.5 rounded-md border border-gray-300 bg-white text-gray-700 text-xs font-medium cursor-pointer transition-all hover:bg-gray-100 hover:border-gray-400"
                        >
                          Clear Selection ({selectedSerialNumbers.size})
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {annotatedImage.annotations.length > 0 && !annotatedImage.loading && (
                <p className="text-xs text-gray-500 mb-4 px-3 py-2 bg-sky-50 rounded border-l-[3px] border-l-blue-500">
                  Click category badges to filter by type. Click rows or boxes to select specific items.
                  {figmaImageUrl && ' Enable Split View to compare with Figma design.'}
                </p>
              )}

              {annotatedImage.loading && (
                <div className="flex flex-col items-center justify-center py-12 text-gray-500">
                  <span className="w-8 h-8 border-[3px] border-violet-500/20 rounded-full border-t-violet-500 animate-spin mb-4" />
                  <p className="text-sm">Loading screenshot and annotations...</p>
                </div>
              )}

              {annotatedImage.error && (
                <div className="mt-4 px-4 py-3 bg-red-50 border border-red-200 rounded-md text-red-600 text-sm">
                  {annotatedImage.error}
                </div>
              )}

              {annotatedImage.baseImageUrl && (
                <div
                  ref={containerRef}
                  className={`flex gap-4 w-full ${splitView ? 'flex-row items-start gap-6' : 'flex-col'}`}
                >
                  {/* Figma Screenshot (split view) */}
                  {splitView && figmaImageUrl && (
                    <div className="flex-1 min-w-0 flex flex-col max-w-[50%]">
                      <div className="flex items-center justify-between px-3 py-2 bg-slate-50 rounded-t-md border border-gray-200 border-b-0">
                        <span className="text-[0.8125rem] font-semibold text-violet-700">Figma Design</span>
                      </div>
                      <div className="relative inline-block max-w-full">
                        <img
                          src={figmaImageUrl}
                          alt="Figma design screenshot"
                          className="max-w-full h-auto border border-gray-200 rounded-b-md shadow-md block"
                        />
                      </div>
                    </div>
                  )}

                  {/* Web Screenshot */}
                  <div className={`flex-1 min-w-0 flex flex-col ${splitView ? 'max-w-[50%]' : 'max-w-full'}`}>
                    {splitView && (
                      <div className="flex items-center justify-between px-3 py-2 bg-slate-50 rounded-t-md border border-gray-200 border-b-0">
                        <span className="text-[0.8125rem] font-semibold text-emerald-700">Web Implementation</span>
                        <span className="text-xs text-gray-500">
                          {getFilteredAnnotations().length} differences
                        </span>
                      </div>
                    )}
                    <div className="relative inline-block max-w-full">
                      <img
                        ref={imageRef}
                        src={annotatedImage.baseImageUrl}
                        alt="Web page screenshot"
                        className={`max-w-full h-auto border border-gray-200 shadow-md block ${splitView ? 'rounded-b-md' : 'rounded-md'}`}
                        onLoad={handleImageLoad}
                      />
                      {/* SVG Overlay */}
                      {annotatedImage.imageWidth > 0 && (
                        <svg
                          className="absolute top-0 left-0 w-full h-full pointer-events-none rounded-md"
                          viewBox={`0 0 ${annotatedImage.imageWidth} ${annotatedImage.imageHeight}`}
                          preserveAspectRatio="xMinYMin meet"
                        >
                          {getFilteredAnnotations().map((ann, idx) => {
                            const color = getAnnotationColor(ann.category)
                            const isHighlighted = isAnnotationHighlighted(ann)
                            const badgeText = ann.serial_numbers.join(',')

                            return (
                              <g
                                key={idx}
                                className="pointer-events-auto cursor-pointer"
                                style={{ transition: 'opacity 0.15s ease' }}
                                onMouseEnter={() => setHoveredSerial(ann.serial_numbers[0])}
                                onMouseLeave={() => setHoveredSerial(null)}
                                onClick={() => ann.serial_numbers.forEach(sn => toggleSerialNumber(sn))}
                              >
                                <rect
                                  x={ann.x}
                                  y={ann.y}
                                  width={ann.width}
                                  height={ann.height}
                                  fill="transparent"
                                  stroke={color}
                                  strokeWidth={isHighlighted ? 4 : 2}
                                  rx={3}
                                  style={{ transition: 'stroke-width 0.15s ease' }}
                                />
                                <g transform={`translate(${ann.x - 2}, ${ann.y - 22})`}>
                                  <rect
                                    x={0}
                                    y={0}
                                    width={Math.max(24, badgeText.length * 8 + 12)}
                                    height={20}
                                    fill={color}
                                    rx={10}
                                    style={{
                                      transition: 'transform 0.15s ease',
                                      ...(isHighlighted ? { transform: 'scale(1.1)', transformOrigin: 'center' } : {})
                                    }}
                                  />
                                  <text
                                    x={Math.max(12, (badgeText.length * 8 + 12) / 2)}
                                    y={14}
                                    fill="white"
                                    fontSize={11}
                                    fontWeight={700}
                                    textAnchor="middle"
                                  >
                                    {badgeText}
                                  </text>
                                </g>
                              </g>
                            )
                          })}
                        </svg>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {annotatedImage.baseImageUrl && (
                <div className="flex gap-3 flex-wrap justify-center mt-4">
                  <a
                    href={annotatedImage.baseImageUrl}
                    download="web_screenshot.png"
                    className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-md text-sm font-medium no-underline transition-colors hover:bg-emerald-600"
                  >
                    Download Web Image
                  </a>
                  {figmaImageUrl && (
                    <a
                      href={figmaImageUrl}
                      download="figma_screenshot.png"
                      className="inline-flex items-center gap-2 px-4 py-2 bg-violet-600 text-white rounded-md text-sm font-medium no-underline transition-colors hover:bg-violet-700"
                    >
                      Download Figma Image
                    </a>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {/* Results Section */}
        {result && (
          <section className="flex-1 flex flex-col min-h-0 animate-fadeIn">
            <div className="flex-1 flex flex-col bg-white rounded-lg border border-gray-200 shadow-sm min-h-0 overflow-hidden">
              {/* Results Header */}
              <div className="shrink-0 flex flex-wrap items-center justify-between px-4 md:px-6 py-4 border-b border-gray-200 gap-4 bg-white">
                <h2 className="text-base font-semibold text-slate-800">Comparison Results</h2>
                <div className="flex flex-wrap gap-2">
                  <span
                    onClick={() => handleCategoryClick('total')}
                    className={`px-3 py-1.5 rounded-full text-xs font-semibold inline-flex items-center gap-1 cursor-pointer transition-all select-none bg-slate-800 text-white hover:scale-105 hover:shadow-lg ${
                      selectedCategory === 'total' ? 'ring-2 ring-slate-800 scale-105 shadow-lg' : ''
                    }`}
                  >
                    Total: {getTotalDifferences()}
                  </span>
                  {Object.entries(getCategoryCounts()).map(([category, count]) =>
                    count > 0 ? (
                      <span
                        key={category}
                        onClick={() => handleCategoryClick(category)}
                        className={`px-3 py-1.5 rounded-full text-xs font-semibold inline-flex items-center gap-1 cursor-pointer transition-all select-none hover:scale-105 hover:shadow-lg ${
                          CATEGORY_CONFIG[category]?.badgeClass || CATEGORY_CONFIG.other.badgeClass
                        } ${selectedCategory === category ? 'ring-2 ring-slate-800 scale-105 shadow-lg' : ''}`}
                      >
                        {CATEGORY_CONFIG[category]?.label || category}: {count}
                      </span>
                    ) : null
                  )}
                </div>
              </div>

              {/* Table */}
              <div className="flex-1 overflow-auto min-h-0">
                <table className="w-full border-collapse text-[0.8125rem]">
                  <thead className="sticky top-0 z-10">
                    <tr>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide w-[50px] text-center">
                        #
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Category
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Sub Category
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Text
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Figma Value
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Web Value
                      </th>
                      <th className="bg-slate-50 px-4 py-3 text-left font-semibold text-slate-500 border-b border-gray-200 whitespace-nowrap text-xs uppercase tracking-wide">
                        Delta
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {getFilteredItems().length > 0 ? (
                      getFilteredItems().map(({ category, item }, index) => {
                        const serialNum = item.serial_number || index + 1
                        const isSelected = selectedSerialNumbers.has(serialNum)
                        const isHovered = hoveredSerial === serialNum
                        const config = getCategoryConfig(category)

                        return (
                          <tr
                            key={index}
                            className={`transition-all cursor-pointer ${config.rowBorder} ${
                              isSelected
                                ? 'bg-indigo-100 hover:bg-indigo-200'
                                : isHovered
                                  ? 'bg-sky-50 ring-inset ring-2 ring-indigo-500'
                                  : 'hover:bg-slate-50'
                            }`}
                            onMouseEnter={() => setHoveredSerial(serialNum)}
                            onMouseLeave={() => setHoveredSerial(null)}
                            onClick={() => toggleSerialNumber(serialNum)}
                          >
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle text-center">
                              <span
                                className={`inline-flex items-center justify-center min-w-[24px] h-6 px-1.5 rounded-full text-xs font-semibold transition-all ${
                                  isSelected
                                    ? 'bg-indigo-600 text-white shadow-md'
                                    : 'bg-indigo-100 text-indigo-600'
                                }`}
                              >
                                {serialNum}
                              </span>
                            </td>
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle">
                              <span
                                className={`inline-block px-2 py-1 rounded text-[0.6875rem] font-semibold capitalize ${config.tagBg} ${config.tagText}`}
                              >
                                {category}
                              </span>
                            </td>
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle">
                              {item.sub_type}
                            </td>
                            <td
                              className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle max-w-[200px] overflow-hidden text-ellipsis whitespace-nowrap"
                              title={item.text}
                            >
                              {item.text}
                            </td>
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle">
                              {String(item.figma_value)}
                            </td>
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle">
                              {String(item.web_value)}
                            </td>
                            <td className="px-4 py-3 border-b border-slate-100 text-gray-700 align-middle font-mono text-xs text-gray-500">
                              {item.delta}
                            </td>
                          </tr>
                        )
                      })
                    ) : (
                      <tr>
                        <td colSpan={7} className="text-center text-gray-400 italic py-12">
                          {selectedCategory ? 'No differences in this category' : 'No differences found'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Tailwind animation keyframe via inline style tag */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease;
        }
      `}</style>
    </div>
  )
}

export default App
