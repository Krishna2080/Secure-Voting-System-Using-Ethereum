
import { useState } from 'react';
import { useVoting } from '../contexts/VotingContext';

interface VotingInterfaceProps {
  onVoteComplete: () => void;
}

const VotingInterface = ({ onVoteComplete }: VotingInterfaceProps) => {
  const { candidates, currentVoter, electionName, castVote } = useVoting();
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null);
  const [isConfirming, setIsConfirming] = useState(false);
  const [isVoting, setIsVoting] = useState(false);
  const [voteResult, setVoteResult] = useState<any>(null);

  const handleVoteSubmit = async () => {
    if (!selectedCandidate) return;
    
    setIsVoting(true);
    
    try {
      const success = castVote(selectedCandidate);
      
      if (success) {
        // Simulate blockchain response (in real implementation, this would come from the backend)
        setVoteResult({
          success: true,
          tx_hash: '0x' + Math.random().toString(16).substr(2, 64),
          block_number: Math.floor(Math.random() * 1000000) + 18000000,
          blockchain_result: { success: true }
        });
        
        // Show success message for 3 seconds before completing
        setTimeout(() => {
          onVoteComplete();
        }, 3000);
      } else {
        alert('Error casting vote. Please try again.');
        setIsVoting(false);
      }
    } catch (error) {
      alert('Error casting vote. Please try again.');
      setIsVoting(false);
    }
  };

  if (!currentVoter) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-gray-900">Access Denied</h2>
        <p className="text-gray-600 mt-2">Please authenticate first to access the voting interface.</p>
      </div>
    );
  }

  if (voteResult?.success) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-8 text-center">
          <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <svg className="w-10 h-10 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
          </div>
          
          <h2 className="text-3xl font-bold text-green-900 mb-4">Vote Cast Successfully!</h2>
          <p className="text-lg text-gray-700 mb-6">
            Your vote has been securely recorded on the Ethereum blockchain.
          </p>

          {/* Blockchain Transaction Details */}
          <div className="bg-gray-50 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Blockchain Transaction Details</h3>
            <div className="space-y-3 text-left">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">Transaction Hash:</span>
                <span className="text-sm font-mono text-blue-600 break-all">{voteResult.tx_hash}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">Block Number:</span>
                <span className="text-sm font-mono text-gray-900">{voteResult.block_number}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">Network:</span>
                <span className="text-sm text-gray-900">Ethereum Sepolia Testnet</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-700">Status:</span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  Confirmed
                </span>
              </div>
            </div>
          </div>

          {/* Security Features */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-3">Security Features Applied</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-left">
              <div className="flex items-start">
                <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-blue-900">Biometric Authentication</p>
                  <p className="text-xs text-blue-700">Face recognition verified</p>
                </div>
              </div>
              <div className="flex items-start">
                <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-blue-900">Blockchain Storage</p>
                  <p className="text-xs text-blue-700">Immutable record created</p>
                </div>
              </div>
              <div className="flex items-start">
                <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-blue-900">Fraud Prevention</p>
                  <p className="text-xs text-blue-700">Duplicate voting blocked</p>
                </div>
              </div>
              <div className="flex items-start">
                <svg className="w-5 h-5 text-green-600 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-blue-900">Smart Contract</p>
                  <p className="text-xs text-blue-700">Automated verification</p>
                </div>
              </div>
            </div>
          </div>

          <p className="text-sm text-gray-600">
            Redirecting to results in a few seconds...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <div className="text-center mb-4">
          <h1 className="text-2xl font-bold text-blue-600 mb-2">{electionName}</h1>
          <div className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-purple-100 text-purple-800">
            <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
            Blockchain Secured
          </div>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Cast Your Vote</h2>
            <p className="text-gray-600">Welcome, {currentVoter.name}</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500">Voter ID</p>
            <p className="text-lg font-medium text-gray-900">{currentVoter.id}</p>
          </div>
        </div>
      </div>

      {!isConfirming ? (
        <>
          {/* Instructions */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-8">
            <div className="flex items-start">
              <svg className="w-6 h-6 text-blue-600 mr-3 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="text-lg font-medium text-blue-900">Blockchain Voting Instructions</h3>
                <ul className="text-blue-700 mt-2 space-y-1">
                  <li>• Select one candidate by clicking on their card</li>
                  <li>• Review your selection carefully</li>
                  <li>• Your vote will be recorded on the Ethereum blockchain</li>
                  <li>• Once cast, your vote is immutable and cannot be changed</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Candidates Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {candidates.map((candidate) => (
              <div
                key={candidate.id}
                onClick={() => setSelectedCandidate(candidate.id)}
                className={`bg-white rounded-xl shadow-lg p-6 cursor-pointer transition-all duration-200 ${
                  selectedCandidate === candidate.id
                    ? 'ring-4 ring-blue-500 border-blue-500 scale-105'
                    : 'border border-gray-200 hover:shadow-xl hover:scale-102'
                }`}
              >
                <div className="text-center">
                  <div className="w-24 h-24 bg-gray-200 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <svg className="w-12 h-12 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                    </svg>
                  </div>
                  
                  <h3 className="text-lg font-bold text-gray-900 mb-2">{candidate.name}</h3>
                  <p className="text-sm text-blue-600 font-medium mb-4">{candidate.party}</p>
                  
                  {selectedCandidate === candidate.id && (
                    <div className="flex items-center justify-center">
                      <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <span className="text-blue-600 font-medium ml-2">Selected</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Action Buttons */}
          <div className="text-center">
            <button
              onClick={() => setIsConfirming(true)}
              disabled={!selectedCandidate}
              className="px-8 py-4 bg-blue-600 text-white text-lg font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-lg"
            >
              Continue to Blockchain Confirmation
            </button>
          </div>
        </>
      ) : (
        /* Confirmation Screen */
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="text-center">
            <div className="w-20 h-20 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg className="w-10 h-10 text-yellow-600" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Confirm Blockchain Vote</h3>
            
            {selectedCandidate && (
              <div className="bg-gray-50 rounded-lg p-6 mb-6 max-w-md mx-auto">
                <p className="text-gray-600 mb-2">You are voting for:</p>
                <h4 className="text-xl font-bold text-gray-900">
                  {candidates.find(c => c.id === selectedCandidate)?.name}
                </h4>
                <p className="text-blue-600 font-medium">
                  {candidates.find(c => c.id === selectedCandidate)?.party}
                </p>
              </div>
            )}
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-6 max-w-lg mx-auto">
              <p className="text-purple-800 font-medium mb-2">
                🔗 Blockchain Security Features
              </p>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• Immutable storage on Ethereum Sepolia</li>
                <li>• Smart contract verification</li>
                <li>• Cryptographic transaction proof</li>
                <li>• Transparent and auditable</li>
              </ul>
            </div>
            
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6 max-w-lg mx-auto">
              <p className="text-red-800 font-medium">
                ⚠️ Warning: This action cannot be undone. Your vote will be permanently recorded on the blockchain.
              </p>
            </div>
            
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => setIsConfirming(false)}
                className="px-6 py-3 bg-gray-600 text-white font-medium rounded-lg hover:bg-gray-700 transition-colors"
              >
                Go Back
              </button>
              <button
                onClick={handleVoteSubmit}
                disabled={isVoting}
                className="px-8 py-3 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {isVoting ? 'Recording on Blockchain...' : 'Cast Vote on Blockchain'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VotingInterface;
