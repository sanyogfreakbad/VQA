import { useState, useRef, useCallback } from 'react'
import './App.css'

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

interface FigmaImageState {
  imageUrl: string | null
  annotations: Annotation[]
  loading: boolean
  error: string | null
  imageWidth: number
  imageHeight: number
}

const CATEGORY_CONFIG: Record<string, { label: string; class: string }> = {
  text: { label: 'Text', class: 'category-text' },
  spacing: { label: 'Spacing', class: 'category-spacing' },
  size: { label: 'Size', class: 'category-size' },
  missing_elements: { label: 'Missing', class: 'category-missing' },
  padding: { label: 'Padding', class: 'category-other' },
  color: { label: 'Color', class: 'category-color' },
  components: { label: 'Components', class: 'category-components' },
  buttons_cta: { label: 'Buttons', class: 'category-buttons' },
  other: { label: 'Other', class: 'category-other' }
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
  const [figmaImage, setFigmaImage] = useState<FigmaImageState>({
    imageUrl: null,
    annotations: [],
    loading: false,
    error: null,
    imageWidth: 0,
    imageHeight: 0
  })
  const [splitView, setSplitView] = useState(false)
  const imageRef = useRef<HTMLImageElement>(null)
  const figmaImageRef = useRef<HTMLImageElement>(null)
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
    setFigmaImage({
      imageUrl: null,
      annotations: [],
      loading: false,
      error: null,
      imageWidth: 0,
      imageHeight: 0
    })

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
        headers: {
          'Content-Type': 'application/json'
        },
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
      // First, get the annotation metadata (without generating a new screenshot)
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
      
      // Use the web screenshot from comparison results if available
      let baseImageUrl = result.web_screenshot_url
      
      // If no web screenshot, we need to capture one via the annotate endpoint (with empty filters to get base)
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

      // Also fetch Figma annotation metadata for missing elements
      const figmaMetadataResponse = await fetch('/api/annotate/figma-metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comparison_results: result })
      })

      if (figmaMetadataResponse.ok) {
        const figmaMetadata = await figmaMetadataResponse.json()
        setFigmaImage({
          imageUrl: result.figma_screenshot_url || null,
          annotations: figmaMetadata.annotations || [],
          loading: false,
          error: null,
          imageWidth: 0,
          imageHeight: 0
        })
      }
    } catch (err) {
      setAnnotatedImage(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to load annotated image'
      }))
    }
  }

  // Handle image load to get dimensions
  const handleImageLoad = useCallback(() => {
    if (imageRef.current) {
      setAnnotatedImage(prev => ({
        ...prev,
        imageWidth: imageRef.current?.naturalWidth || 0,
        imageHeight: imageRef.current?.naturalHeight || 0
      }))
    }
  }, [])

  // Handle Figma image load to get dimensions
  const handleFigmaImageLoad = useCallback(() => {
    if (figmaImageRef.current) {
      setFigmaImage(prev => ({
        ...prev,
        imageWidth: figmaImageRef.current?.naturalWidth || 0,
        imageHeight: figmaImageRef.current?.naturalHeight || 0
      }))
    }
  }, [])

  // Toggle serial number selection
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

  // Clear all serial number selections
  const clearSerialSelection = () => {
    setSelectedSerialNumbers(new Set())
  }

  // Get filtered annotations based on category and serial number selection
  const getFilteredAnnotations = useCallback((): Annotation[] => {
    let filtered = annotatedImage.annotations

    // Filter by category
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

    // Filter by selected serial numbers (if any selected)
    if (selectedSerialNumbers.size > 0) {
      filtered = filtered.filter(ann => 
        ann.serial_numbers.some(sn => selectedSerialNumbers.has(sn))
      )
    }

    return filtered
  }, [annotatedImage.annotations, selectedCategory, selectedSerialNumbers])

  // Get filtered Figma annotations (missing elements only)
  const getFilteredFigmaAnnotations = useCallback((): Annotation[] => {
    let filtered = figmaImage.annotations

    // Show missing elements only when:
    // 1. No category filter (show all)
    // 2. 'total' is selected
    // 3. 'missing_elements' is selected
    if (selectedCategory && selectedCategory !== 'total' && selectedCategory !== 'missing_elements') {
      return [] // Hide Figma annotations when other categories are selected
    }

    // Filter by selected serial numbers (if any selected)
    if (selectedSerialNumbers.size > 0) {
      filtered = filtered.filter(ann => 
        ann.serial_numbers.some(sn => selectedSerialNumbers.has(sn))
      )
    }

    return filtered
  }, [figmaImage.annotations, selectedCategory, selectedSerialNumbers])

  // Get color for annotation category
  const getAnnotationColor = (category: string): string => {
    const colors = annotatedImage.categoryColors[category]
    return colors?.border || '#f59e0b'
  }

  // Check if an annotation should be highlighted (hovered)
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

  const getCategoryClass = (category: string): string => {
    const normalizedCategory = category.toLowerCase().replace(/\s+/g, '_')
    return CATEGORY_CONFIG[normalizedCategory]?.class || 'category-other'
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <div>
            <h1>Figma vs Web Comparator</h1>
            <p>Compare your Figma designs with live web implementations</p>
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="form-section">
          <div className="form-card">
            <h2>Configuration</h2>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="figmaUrl">Figma URL *</label>
                <input
                  type="url"
                  id="figmaUrl"
                  name="figmaUrl"
                  value={formData.figmaUrl}
                  onChange={handleInputChange}
                  placeholder="https://www.figma.com/design/..."
                />
              </div>

              <div className="form-group">
                <label htmlFor="webUrl">Web URL *</label>
                <input
                  type="url"
                  id="webUrl"
                  name="webUrl"
                  value={formData.webUrl}
                  onChange={handleInputChange}
                  placeholder="https://example.com/page"
                />
              </div>

              <div className="form-group">
                <label htmlFor="loginUrl">Login URL</label>
                <input
                  type="url"
                  id="loginUrl"
                  name="loginUrl"
                  value={formData.loginUrl}
                  onChange={handleInputChange}
                  placeholder="https://example.com/login"
                />
              </div>
            </div>

            <div className="credentials-section">
              <h3>Credentials (optional)</h3>
              <div className="credentials-grid">
                <div className="form-group">
                  <label htmlFor="username">Username</label>
                  <input
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    placeholder="Enter username"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="password">Password</label>
                  <input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    placeholder="Enter password"
                  />
                </div>
              </div>
            </div>

            <div className="options-section">
              <h3>Options</h3>
              <div className="checkbox-group">
                <input
                  type="checkbox"
                  id="useGemini"
                  name="useGemini"
                  checked={formData.useGemini}
                  onChange={handleInputChange}
                />
                <label htmlFor="useGemini">
                  Use Gemini AI Refinement
                  <span className="option-description">Enable AI-powered visual validation to reduce false positives</span>
                </label>
              </div>
            </div>

            <div className="form-actions">
              <button 
                className="compare-btn"
                onClick={handleCompare}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Comparing...
                  </>
                ) : (
                  'Compare'
                )}
              </button>
              {result && (
                <button 
                  className="show-image-btn"
                  onClick={handleShowImage}
                  disabled={annotatedImage.loading}
                >
                  {annotatedImage.loading ? (
                    <>
                      <span className="spinner"></span>
                      Generating...
                    </>
                  ) : (
                    'Show Image'
                  )}
                </button>
              )}
            </div>

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Annotated Image Section with Dynamic SVG Overlay */}
        {(annotatedImage.baseImageUrl || annotatedImage.loading || annotatedImage.error) && (
          <section className="annotated-image-section">
            <div className="annotated-image-card">
              <div className="annotated-image-header">
                <h2>Annotated Screenshot</h2>
                <div className="header-controls">
                  {figmaImage.imageUrl && (
                    <button 
                      className={`split-view-btn ${splitView ? 'active' : ''}`}
                      onClick={() => setSplitView(!splitView)}
                    >
                      {splitView ? 'Single View' : 'Split View'}
                    </button>
                  )}
                  {annotatedImage.annotations.length > 0 && (
                    <div className="annotation-controls">
                      <span className="annotation-count">
                        Showing {getFilteredAnnotations().length} of {annotatedImage.annotations.length} boxes
                      </span>
                      {selectedSerialNumbers.size > 0 && (
                        <button 
                          className="clear-selection-btn"
                          onClick={clearSerialSelection}
                        >
                          Clear Selection ({selectedSerialNumbers.size})
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
              {annotatedImage.annotations.length > 0 && !annotatedImage.loading && (
                <p className="annotation-tip">
                  Click category badges to filter by type. Click rows or boxes to select specific items.
                  {figmaImage.imageUrl && ' Enable Split View to see Figma design with missing elements.'}
                </p>
              )}
              {annotatedImage.loading && (
                <div className="image-loading">
                  <span className="spinner"></span>
                  <p>Loading screenshot and annotations...</p>
                </div>
              )}
              {annotatedImage.error && (
                <div className="error-message">
                  {annotatedImage.error}
                </div>
              )}
              {annotatedImage.baseImageUrl && (
                <div className={`images-container ${splitView ? 'split-view' : ''}`} ref={containerRef}>
                  {/* Figma Screenshot (shown in split view) */}
                  {splitView && figmaImage.imageUrl && (
                    <div className="image-panel figma-panel">
                      <div className="panel-header">
                        <span className="panel-label">Figma Design</span>
                        <span className="panel-count">
                          {getFilteredFigmaAnnotations().length} missing elements
                        </span>
                      </div>
                      <div className="image-wrapper">
                        <img 
                          ref={figmaImageRef}
                          src={figmaImage.imageUrl} 
                          alt="Figma design screenshot" 
                          className="annotated-image"
                          onLoad={handleFigmaImageLoad}
                        />
                        {/* SVG Overlay for Figma missing elements */}
                        {figmaImage.imageWidth > 0 && (
                          <svg 
                            className="annotation-overlay"
                            viewBox={`0 0 ${figmaImage.imageWidth} ${figmaImage.imageHeight}`}
                            preserveAspectRatio="xMinYMin meet"
                          >
                            {getFilteredFigmaAnnotations().map((ann, idx) => {
                              const color = annotatedImage.categoryColors['missing_elements']?.border || '#ef4444'
                              const isHighlighted = isAnnotationHighlighted(ann)
                              const badgeText = ann.serial_numbers.join(',')
                              
                              return (
                                <g 
                                  key={idx} 
                                  className={`annotation-group ${isHighlighted ? 'highlighted' : ''}`}
                                  onMouseEnter={() => setHoveredSerial(ann.serial_numbers[0])}
                                  onMouseLeave={() => setHoveredSerial(null)}
                                  onClick={() => ann.serial_numbers.forEach(sn => toggleSerialNumber(sn))}
                                  style={{ cursor: 'pointer' }}
                                >
                                  <rect
                                    x={ann.x}
                                    y={ann.y}
                                    width={ann.width}
                                    height={ann.height}
                                    fill="rgba(239, 68, 68, 0.1)"
                                    stroke={color}
                                    strokeWidth={isHighlighted ? 4 : 2}
                                    strokeDasharray={isHighlighted ? "0" : "5,5"}
                                    rx={3}
                                    className="annotation-box"
                                  />
                                  <g transform={`translate(${ann.x - 2}, ${ann.y - 22})`}>
                                    <rect
                                      x={0}
                                      y={0}
                                      width={Math.max(24, badgeText.length * 8 + 12)}
                                      height={20}
                                      fill={color}
                                      rx={10}
                                      className="annotation-badge-bg"
                                    />
                                    <text
                                      x={Math.max(12, (badgeText.length * 8 + 12) / 2)}
                                      y={14}
                                      fill="white"
                                      fontSize={11}
                                      fontWeight={700}
                                      textAnchor="middle"
                                      className="annotation-badge-text"
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
                  )}

                  {/* Web Screenshot */}
                  <div className={`image-panel web-panel ${splitView ? '' : 'full-width'}`}>
                    {splitView && (
                      <div className="panel-header">
                        <span className="panel-label">Web Implementation</span>
                        <span className="panel-count">
                          {getFilteredAnnotations().length} differences
                        </span>
                      </div>
                    )}
                    <div className="image-wrapper">
                      <img 
                        ref={imageRef}
                        src={annotatedImage.baseImageUrl} 
                        alt="Web page screenshot" 
                        className="annotated-image"
                        onLoad={handleImageLoad}
                      />
                      {/* SVG Overlay for dynamic annotations */}
                      {annotatedImage.imageWidth > 0 && (
                        <svg 
                          className="annotation-overlay"
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
                                className={`annotation-group ${isHighlighted ? 'highlighted' : ''}`}
                                onMouseEnter={() => setHoveredSerial(ann.serial_numbers[0])}
                                onMouseLeave={() => setHoveredSerial(null)}
                                onClick={() => ann.serial_numbers.forEach(sn => toggleSerialNumber(sn))}
                                style={{ cursor: 'pointer' }}
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
                                  className="annotation-box"
                                />
                                <g transform={`translate(${ann.x - 2}, ${ann.y - 22})`}>
                                  <rect
                                    x={0}
                                    y={0}
                                    width={Math.max(24, badgeText.length * 8 + 12)}
                                    height={20}
                                    fill={color}
                                    rx={10}
                                    className="annotation-badge-bg"
                                  />
                                  <text
                                    x={Math.max(12, (badgeText.length * 8 + 12) / 2)}
                                    y={14}
                                    fill="white"
                                    fontSize={11}
                                    fontWeight={700}
                                    textAnchor="middle"
                                    className="annotation-badge-text"
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
                <div className="image-actions">
                  <a 
                    href={annotatedImage.baseImageUrl} 
                    download="web_screenshot.png"
                    className="download-btn"
                  >
                    Download Web Image
                  </a>
                  {figmaImage.imageUrl && (
                    <a 
                      href={figmaImage.imageUrl} 
                      download="figma_screenshot.png"
                      className="download-btn download-figma"
                    >
                      Download Figma Image
                    </a>
                  )}
                </div>
              )}
            </div>
          </section>
        )}

        {result && (
          <section className="results-section">
            <div className="results-card">
              <div className="results-header">
                <h2>Comparison Results</h2>
                <div className="summary-badges">
                  <span 
                    className={`badge badge-total ${selectedCategory === 'total' ? 'badge-selected' : ''}`}
                    onClick={() => handleCategoryClick('total')}
                  >
                    Total: {getTotalDifferences()}
                  </span>
                  {Object.entries(getCategoryCounts()).map(([category, count]) => (
                    count > 0 && (
                      <span 
                        key={category} 
                        className={`badge badge-${category} ${selectedCategory === category ? 'badge-selected' : ''}`}
                        onClick={() => handleCategoryClick(category)}
                      >
                        {CATEGORY_CONFIG[category]?.label || category}: {count}
                      </span>
                    )
                  ))}
                </div>
              </div>

              <div className="table-container">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th className="serial-col">#</th>
                      <th>Category</th>
                      <th>Sub Category</th>
                      <th>Text</th>
                      <th>Figma Value</th>
                      <th>Web Value</th>
                      <th>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getFilteredItems().length > 0 ? (
                      getFilteredItems().map(({ category, item }, index) => {
                        const serialNum = item.serial_number || index + 1
                        const isSelected = selectedSerialNumbers.has(serialNum)
                        const isHovered = hoveredSerial === serialNum
                        
                        return (
                          <tr 
                            key={index} 
                            className={`${getCategoryClass(category)} ${isSelected ? 'row-selected' : ''} ${isHovered ? 'row-hovered' : ''}`}
                            onMouseEnter={() => setHoveredSerial(serialNum)}
                            onMouseLeave={() => setHoveredSerial(null)}
                            onClick={() => toggleSerialNumber(serialNum)}
                            style={{ cursor: 'pointer' }}
                          >
                            <td className="serial-cell">
                              <span className={`serial-badge ${isSelected ? 'selected' : ''}`}>
                                {serialNum}
                              </span>
                            </td>
                            <td>
                              <span className={`category-tag ${getCategoryClass(category)}`}>
                                {category}
                              </span>
                            </td>
                            <td>{item.sub_type}</td>
                            <td className="text-cell" title={item.text}>{item.text}</td>
                            <td>{String(item.figma_value)}</td>
                            <td>{String(item.web_value)}</td>
                            <td className="delta-cell">{item.delta}</td>
                          </tr>
                        )
                      })
                    ) : (
                      <tr>
                        <td colSpan={7} className="no-data">
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
    </div>
  )
}

export default App
