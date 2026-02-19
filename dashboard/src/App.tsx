import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WardHeatmap } from './components/WardHeatmap';
import { PatientTimeline } from './components/PatientTimeline';
import { PharmacyView, InfectionControlView } from './components/PharmacyView';

const queryClient = new QueryClient();

const Dashboard = () => (
  <div className="p-6">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">ResisTrack Dashboard</h1>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Link to="/heatmap" className="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold text-blue-600">Ward Heatmap</h2>
        <p className="text-gray-600 mt-2">View AMR risk levels across all hospital wards</p>
      </Link>
      <Link to="/pharmacy" className="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold text-blue-600">Pharmacy View</h2>
        <p className="text-gray-600 mt-2">Review high-risk patients and antibiotic recommendations</p>
      </Link>
      <Link to="/infection-control" className="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold text-blue-600">Infection Control</h2>
        <p className="text-gray-600 mt-2">Monitor outbreaks and MDRO cluster trends</p>
      </Link>
      <Link to="/reports" className="block p-6 bg-white rounded-lg shadow hover:shadow-md transition-shadow">
        <h2 className="text-xl font-semibold text-blue-600">Stewardship Reports</h2>
        <p className="text-gray-600 mt-2">Download weekly AMR stewardship reports</p>
      </Link>
    </div>
  </div>
);

const Reports = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Stewardship Reports</h1>
    <p className="text-gray-600">Weekly AMR stewardship reports will be available for download here.</p>
  </div>
);

const Settings = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold text-gray-900 mb-4">Settings</h1>
    <p className="text-gray-600">Application settings and user preferences.</p>
  </div>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 py-3">
              <div className="flex items-center justify-between">
                <Link to="/" className="text-xl font-bold text-blue-700">ResisTrack</Link>
                <ul className="flex space-x-6">
                  <li><Link to="/" className="text-gray-700 hover:text-blue-600 transition-colors">Dashboard</Link></li>
                  <li><Link to="/heatmap" className="text-gray-700 hover:text-blue-600 transition-colors">Heatmap</Link></li>
                  <li><Link to="/pharmacy" className="text-gray-700 hover:text-blue-600 transition-colors">Pharmacy</Link></li>
                  <li><Link to="/infection-control" className="text-gray-700 hover:text-blue-600 transition-colors">IC</Link></li>
                  <li><Link to="/reports" className="text-gray-700 hover:text-blue-600 transition-colors">Reports</Link></li>
                  <li><Link to="/settings" className="text-gray-700 hover:text-blue-600 transition-colors">Settings</Link></li>
                </ul>
              </div>
            </div>
          </nav>

          <main className="max-w-7xl mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/patient/:id" element={<PatientTimeline />} />
              <Route path="/heatmap" element={<WardHeatmap />} />
              <Route path="/pharmacy" element={<PharmacyView />} />
              <Route path="/infection-control" element={<InfectionControlView />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
