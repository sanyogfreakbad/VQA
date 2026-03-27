import { useState } from 'react'
import './App.css'

interface WebPosition {
  x: number
  y: number
  width: number
  height: number
}

interface ComparisonItem {
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
}

interface FormData {
  figmaUrl: string
  webUrl: string
  loginUrl: string
  username: string
  password: string
  useGemini: boolean
}

interface AnnotatedImageState {
  url: string | null
  loading: boolean
  error: string | null
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
  const [annotatedImage, setAnnotatedImage] = useState<AnnotatedImageState>({
    url: null,
    loading: false,
    error: null
  })

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
    setAnnotatedImage({ url: null, loading: false, error: null })

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

    setAnnotatedImage({ url: null, loading: true, error: null })

    const requestBody = {
      comparison_results: result,
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
      viewport: { width: 1440, height: 800 }
    }

    try {
      const response = await fetch('/api/annotate', {
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
      setAnnotatedImage({
        url: data.annotated_screenshot_url,
        loading: false,
        error: null
      })
    } catch (err) {
      setAnnotatedImage({
        url: null,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to generate annotated image'
      })
    }
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

        {/* Annotated Image Section */}
        {(annotatedImage.url || annotatedImage.loading || annotatedImage.error) && (
          <section className="annotated-image-section">
            <div className="annotated-image-card">
              <h2>Annotated Screenshot</h2>
              {annotatedImage.loading && (
                <div className="image-loading">
                  <span className="spinner"></span>
                  <p>Generating annotated screenshot...</p>
                </div>
              )}
              {annotatedImage.error && (
                <div className="error-message">
                  {annotatedImage.error}
                </div>
              )}
              {annotatedImage.url && (
                <div className="annotated-image-container">
                  <img 
                    src={annotatedImage.url} 
                    alt="Annotated screenshot showing differences" 
                    className="annotated-image"
                  />
                  <a 
                    href={annotatedImage.url} 
                    download="annotated_screenshot.png"
                    className="download-btn"
                  >
                    Download Image
                  </a>
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
                      getFilteredItems().map(({ category, item }, index) => (
                        <tr key={index} className={getCategoryClass(category)}>
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
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="no-data">
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
