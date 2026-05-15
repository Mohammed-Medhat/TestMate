import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import MetricsPanel from './components/MetricsPanel'
import Breadcrumb from './components/Breadcrumb'
import SettingsModal from './components/SettingsModal'
import UploadTab from './components/UploadTab'
import RequirementsTab from './components/RequirementsTab'
import TestCasesTab from './components/TestCasesTab'
import ReadmeTab from './components/ReadmeTab'

export default function App() {
  const [showSettings, setShowSettings] = useState(false)
  const [activeTab, setActiveTab] = useState('upload')
  const [requirements, setRequirements] = useState(null)

  function handleViewRequirements(reqs) {
    setRequirements(reqs)
    setActiveTab('requirements')
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {/* ── Left Sidebar: icon rail + file tree ── */}
      <Sidebar onSettingsClick={() => setShowSettings(true)} />

      {/* ── Center: breadcrumb + main content ── */}
      <div className="flex flex-1 flex-col overflow-hidden border-x border-zinc-800">
        <Breadcrumb
          activeTab={activeTab}
          onTabChange={setActiveTab}
          onSettingsClick={() => setShowSettings(true)}
        />
        <main className="flex-1 overflow-auto p-6">
          {activeTab === 'upload' && <UploadTab onViewRequirements={handleViewRequirements} />}
          {activeTab === 'requirements' && <RequirementsTab requirements={requirements} />}
          {activeTab === 'testcases' && <TestCasesTab />}
          {activeTab === 'readme' && <ReadmeTab />}
        </main>
      </div>

      {/* ── Right: global metrics panel ── */}
      <MetricsPanel />

      {/* ── Settings modal overlay ── */}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  )
}
