import { useState } from 'react';
import { useVoting } from '../contexts/VotingContext';
import FaceCapture from './FaceCapture';
import VoterRegistration from './VoterRegistration';
import { faceRecognitionApi } from '../services/faceRecognitionApi';

interface VoterAuthProps {
  onAuthenticated: () => void;
}

const VoterAuth = ({ onAuthenticated }: VoterAuthProps) => {
  const { setCurrentVoter, voters, electionName } = useVoting();
  const [currentMode, setCurrentMode] = useState<'select' | 'register' | 'authenticate'>('select');
  const [isScanning, setIsScanning] = useState(false);
  const [authStatus, setAuthStatus] = useState<'idle' | 'success' | 'failed'>('idle');

  const handleFaceCapture = async (result: any) => {
    setAuthStatus('idle');
    
    if (result.success) {
      const voterName = result.voter_name;
      
      if (result.has_voted) {
        setAuthStatus('failed');
        alert(`${voterName}, you have already voted in this election.`);
      } else {
        // Find the voter in local context
        const voter = voters.find(v => v.name === voterName);
        if (voter) {
          setCurrentVoter(voter);
          setAuthStatus('success');
          setTimeout(() => {
            onAuthenticated();
          }, 2000);
        } else {
          setAuthStatus('failed');
          alert('Voter found in backend but not in local database. Please contact administrator.');
        }
      }
    } else {
      setAuthStatus('failed');
      setTimeout(() => {
        setAuthStatus('idle');
      }, 3000);
    }
    
    setIsScanning(false);
  };

  const handleRegistrationComplete = () => {
    setCurrentMode('authenticate');
  };

  if (currentMode === 'register') {
    return <VoterRegistration onRegistrationComplete={handleRegistrationComplete} />;
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="bg-blue-600 text-white rounded-xl p-6 mb-6">
          <h1 className="text-4xl font-bold mb-2">{electionName}</h1>
          <p className="text-blue-100 text-lg">Secure Biometric Voting System</p>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Voter Portal</h2>
        <p className="text-lg text-gray-600">
          Register as a new voter or authenticate with biometric Face detection
        </p>
      </div>

      {currentMode === 'select' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* New Voter Registration */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">New Voter</h3>
              <p className="text-gray-600 mb-4">
                First time voting? Register here
              </p>
              
              <button
                onClick={() => setCurrentMode('register')}
                className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors"
              >
                Register Now
              </button>
            </div>
          </div>

          {/* Existing Voter Authentication */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 8a6 6 0 01-7.743 5.743L10 14l-4 4H4v-4l4.257-4.257A6 6 0 118 8zm-6-2a1 1 0 100 2 2 2 0 012 2 1 1 0 102 0 4 4 0 00-4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-3">Existing Voter</h3>
              <p className="text-gray-600 mb-4">
                Already registered? then Authenticate
              </p>
              
              <button
                onClick={() => setCurrentMode('authenticate')}
                className="inline-flex items-center px-6 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors"
              >
                Authenticate
              </button>
            </div>
          </div>
        </div>
      )}

      {currentMode === 'authenticate' && (
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-bold text-gray-900">Advanced Face Authentication</h3>
            <button
              onClick={() => setCurrentMode('select')}
              className="text-gray-500 hover:text-gray-700 transition-colors"
            >
              ← Back to options
            </button>
          </div>

          {!isScanning ? (
            <div className="text-center space-y-6">
              {/* Authentication Status */}
              {authStatus === 'success' && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                  <div className="flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mx-auto mb-4">
                    <svg className="w-8 h-8 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-green-900 mb-2">Authentication Successful!</h3>
                  <p className="text-green-700">Biometric verification complete. Redirecting to voting interface...</p>
                </div>
              )}

              {authStatus === 'failed' && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                  <div className="flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mx-auto mb-4">
                    <svg className="w-8 h-8 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-red-900 mb-2">Authentication Failed</h3>
                  <p className="text-red-700">Face not recognized or you have already voted. Please try again or register if you're a new voter.</p>
                </div>
              )}

              {authStatus === 'idle' && (
                <>
                  {/* Security Features */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div className="text-center">
                      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <svg className="w-8 h-8 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-900">MTCNN Detection</h3>
                      <p className="text-sm text-gray-600">Multi-task CNN face detection</p>
                    </div>
                    
                    <div className="text-center">
                      <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <svg className="w-8 h-8 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 000-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-900">FaceNet Embeddings</h3>
                      <p className="text-sm text-gray-600">Deep learning face recognition</p>
                    </div>
                    
                    <div className="text-center">
                      <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                        <svg className="w-8 h-8 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-900">Cosine Similarity</h3>
                      <p className="text-sm text-gray-600">High-precision matching algorithm</p>
                    </div>
                  </div>

                  {/* Start Authentication Button */}
                  <button
                    onClick={() => setIsScanning(true)}
                    className="inline-flex items-center px-8 py-4 bg-blue-600 text-white text-lg font-medium rounded-lg hover:bg-blue-700 transition-colors shadow-lg"
                  >
                    <svg className="w-6 h-6 mr-3" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    </svg>
                    Start Advanced Authentication
                  </button>

                  <p className="text-sm text-gray-500 mt-4">
                    Click the button above to begin AI-powered face recognition
                  </p>
                </>
              )}
            </div>
          ) : (
            <FaceCapture
              mode="authenticate"
              onCapture={handleFaceCapture}
              onCancel={() => setIsScanning(false)}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default VoterAuth;
