
import { useState } from 'react';
import AdminAuth from '../components/AdminAuth';
import AdminPanel from '../components/AdminPanel';
import VoterAuth from '../components/VoterAuth';
import VotingInterface from '../components/VotingInterface';
import ResultsDashboard from '../components/ResultsDashboard';
import { VotingProvider } from '../contexts/VotingContext';

const Index = () => {
  const [currentView, setCurrentView] = useState<'auth' | 'admin' | 'vote' | 'results'>('auth');
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState(false);

  const handleAdminAccess = () => {
    if (isAdminAuthenticated) {
      setCurrentView('admin');
    } else {
      setCurrentView('admin');
    }
  };

  const handleAdminAuthenticated = () => {
    setIsAdminAuthenticated(true);
  };

  return (
    <VotingProvider>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Header */}
        <header className="bg-white shadow-lg border-b-4 border-blue-600">
          <div className="max-w-7xl mx-auto px-4 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                  <svg className="w-7 h-7 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gray-900">Secure Voting System</h1>
                  <p className="text-sm text-gray-600">Using Face Detection & Ethereum</p>
                </div>
              </div>
              
              <nav className="flex space-x-4">
                <button
                  onClick={() => setCurrentView('auth')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    currentView === 'auth' 
                      ? 'bg-blue-600 text-white' 
                      : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                  }`}
                >
                  Voter Portal
                </button>
                <button
                  onClick={handleAdminAccess}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    currentView === 'admin' 
                      ? 'bg-red-600 text-white' 
                      : 'text-gray-600 hover:text-red-600 hover:bg-red-50'
                  }`}
                >
                  Admin Panel
                </button>
                <button
                  onClick={() => setCurrentView('results')}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    currentView === 'results' 
                      ? 'bg-blue-600 text-white' 
                      : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                  }`}
                >
                  Results
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 py-8">
          {currentView === 'auth' && <VoterAuth onAuthenticated={() => setCurrentView('vote')} />}
          {currentView === 'admin' && (
            !isAdminAuthenticated ? 
              <AdminAuth onAuthenticated={handleAdminAuthenticated} /> : 
              <AdminPanel />
          )}
          {currentView === 'vote' && <VotingInterface onVoteComplete={() => setCurrentView('results')} />}
          {currentView === 'results' && <ResultsDashboard />}
        </main>

        {/* Footer */}
        <footer className="bg-gray-800 text-white py-8 mt-16">
          <div className="max-w-7xl mx-auto px-4 text-center">
            <p className="text-gray-400">
              © 2025 Secure Voting system using Ethereum
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Developed by Group 17 :Padmanabh,Sanved,Anish & sakshi
            </p>
          </div>
        </footer>
      </div>
    </VotingProvider>
  );
};

export default Index;
