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

interface ComparisonResult {
  by_category: ByCategory
  summary?: {
    total_differences: number
    critical: number
    warning: number
    info: number
  }
}

interface FormData {
  figmaUrl: string
  webUrl: string
  loginUrl: string
  username: string
  password: string
}

function App() {
  const [formData, setFormData] = useState<FormData>({
    figmaUrl: '',
    webUrl: '',
    loginUrl: '',
    username: '',
    password: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ComparisonResult | null>(null)

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
  }

  const handleCompare = async () => {
    if (!formData.figmaUrl || !formData.webUrl) {
      setError('Figma URL and Web URL are required')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

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
      max_depth: 50
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

  const getSeverityColor = (severity: string): string => {
    switch (severity.toLowerCase()) {
      case 'critical':
        return 'severity-critical'
      case 'warning':
        return 'severity-warning'
      case 'info':
        return 'severity-info'
      default:
        return ''
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Figma vs Web Comparator</h1>
        <p>Compare your Figma designs with live web implementations</p>
      </header>

      <main className="main-content">
        <section className="form-section">
          <div className="form-card">
            <h2>Configuration</h2>
            
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
              <label htmlFor="loginUrl">Login URL (optional)</label>
              <input
                type="url"
                id="loginUrl"
                name="loginUrl"
                value={formData.loginUrl}
                onChange={handleInputChange}
                placeholder="https://example.com/login"
              />
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

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
          </div>
        </section>

        {result && (
          <section className="results-section">
            <div className="results-card">
              <div className="results-header">
                <h2>Comparison Results</h2>
                {result.summary && (
                  <div className="summary-badges">
                    <span className="badge badge-total">
                      Total: {result.summary.total_differences}
                    </span>
                    <span className="badge badge-critical">
                      Critical: {result.summary.critical}
                    </span>
                    <span className="badge badge-warning">
                      Warning: {result.summary.warning}
                    </span>
                    <span className="badge badge-info">
                      Info: {result.summary.info}
                    </span>
                  </div>
                )}
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
                    {getAllItems().length > 0 ? (
                      getAllItems().map(({ category, item }, index) => (
                        <tr key={index} className={getSeverityColor(item.severity)}>
                          <td>{category}</td>
                          <td>{item.sub_type}</td>
                          <td className="text-cell">{item.text}</td>
                          <td>{String(item.figma_value)}</td>
                          <td>{String(item.web_value)}</td>
                          <td className="delta-cell">{item.delta}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="no-data">
                          No differences found
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
