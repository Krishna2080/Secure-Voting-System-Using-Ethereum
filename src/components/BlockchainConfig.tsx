
import { useState, useEffect } from 'react';
import { blockchainApi } from '../services/blockchainApi';

interface BlockchainConfigProps {
  onConfigured: () => void;
}

const BlockchainConfig = ({ onConfigured }: BlockchainConfigProps) => {
  const [config, setConfig] = useState({
    contract_address: '',
    rpc_url: 'https://rpc.sepolia.org',
    private_key: '',
    account_address: ''
  });
  
  const [status, setStatus] = useState({
    connected: false,
    contract_configured: false,
    contract_address: ''
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const [metaMaskStatus, setMetaMaskStatus] = useState<'not_installed' | 'disconnected' | 'connected' | 'wrong_network'>('disconnected');

  useEffect(() => {
    loadBlockchainStatus();
    checkMetaMaskStatus();
  }, []);

  const checkMetaMaskStatus = async () => {
    if (typeof window.ethereum === 'undefined') {
      setMetaMaskStatus('not_installed');
      return;
    }

    try {
      const accounts = await window.ethereum.request({ method: 'eth_accounts' });
      if (accounts.length === 0) {
        setMetaMaskStatus('disconnected');
        return;
      }

      const chainId = await window.ethereum.request({ method: 'eth_chainId' });
      if (chainId !== '0xaa36a7') { // Sepolia chainId
        setMetaMaskStatus('wrong_network');
        return;
      }

      setMetaMaskStatus('connected');
      setConfig(prev => ({ ...prev, account_address: accounts[0] }));
    } catch (error) {
      console.error('Error checking MetaMask status:', error);
      setMetaMaskStatus('disconnected');
    }
  };

  const loadBlockchainStatus = async () => {
    try {
      const statusResult = await blockchainApi.getBlockchainStatus();
      setStatus(statusResult);
    } catch (error) {
      console.error('Failed to load blockchain status:', error);
    }
  };

  const connectMetaMask = async () => {
    if (typeof window.ethereum === 'undefined') {
      alert('MetaMask is not installed. Please install MetaMask extension from metamask.io');
      return;
    }

    try {
      // Request account access
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
      
      // Switch to Sepolia network
      try {
        await window.ethereum.request({
          method: 'wallet_switchEthereumChain',
          params: [{ chainId: '0xaa36a7' }], // Sepolia chainId
        });
      } catch (switchError: any) {
        // If Sepolia is not added, add it
        if (switchError.code === 4902) {
          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: '0xaa36a7',
              chainName: 'Sepolia Test Network',
              nativeCurrency: {
                name: 'SepoliaETH',
                symbol: 'ETH',
                decimals: 18
              },
              rpcUrls: ['https://rpc.sepolia.org'],
              blockExplorerUrls: ['https://sepolia.etherscan.io']
            }]
          });
        }
      }
      
      setConfig(prev => ({
        ...prev,
        account_address: accounts[0]
      }));
      
      setMetaMaskStatus('connected');
      alert('MetaMask connected successfully! Please enter your private key and contract address.');
    } catch (error) {
      console.error('Failed to connect MetaMask:', error);
      alert('Failed to connect to MetaMask. Please try again.');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const result = await blockchainApi.configureBlockchain(config);
      
      if (result.success) {
        alert('Blockchain configured successfully!');
        await loadBlockchainStatus();
        onConfigured();
      } else {
        alert(`Configuration failed: ${result.message}`);
      }
    } catch (error) {
      alert('Failed to configure blockchain. Please check your settings.');
    } finally {
      setIsLoading(false);
    }
  };

  const getMetaMaskStatusColor = () => {
    switch (metaMaskStatus) {
      case 'connected': return 'bg-green-500';
      case 'wrong_network': return 'bg-yellow-500';
      case 'not_installed': return 'bg-red-500';
      default: return 'bg-red-500';
    }
  };

  const getMetaMaskStatusText = () => {
    switch (metaMaskStatus) {
      case 'connected': return 'MetaMask Connected (Sepolia)';
      case 'wrong_network': return 'MetaMask Connected (Wrong Network)';
      case 'not_installed': return 'MetaMask Not Installed';
      default: return 'MetaMask Disconnected';
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-xl shadow-lg p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Blockchain Configuration</h2>
        
        {/* Status Display */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Current Status</h3>
          <div className="space-y-2">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-3 ${getMetaMaskStatusColor()}`}></div>
              <span className="text-sm">{getMetaMaskStatusText()}</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-3 ${status.connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">Backend Connection: {status.connected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-3 ${status.contract_configured ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">Smart Contract: {status.contract_configured ? 'Configured' : 'Not Configured'}</span>
            </div>
            {status.contract_address && (
              <div className="text-xs text-gray-600">
                Contract: {status.contract_address}
              </div>
            )}
          </div>
        </div>

        {/* Setup Instructions */}
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">Setup Instructions</h3>
          <ol className="text-sm text-blue-700 space-y-2 list-decimal list-inside">
            <li>Install MetaMask browser extension from metamask.io</li>
            <li>Switch to Sepolia testnet in MetaMask</li>
            <li>Get Sepolia ETH from faucet: sepolia-faucet.pk910.de</li>
            <li>Open Remix IDE (remix.ethereum.org)</li>
            <li>Deploy the SecureVoting contract on Sepolia</li>
            <li>Copy the deployed contract address</li>
            <li>Connect MetaMask below and configure</li>
          </ol>
        </div>

        {/* MetaMask Connection */}
        <div className="mb-6 p-4 bg-orange-50 border border-orange-200 rounded-lg">
          <h3 className="text-lg font-semibold text-orange-900 mb-3">MetaMask Connection</h3>
          <p className="text-orange-700 mb-3">
            Connect your MetaMask wallet to automatically configure Sepolia testnet settings.
          </p>
          <button
            type="button"
            onClick={connectMetaMask}
            disabled={metaMaskStatus === 'not_installed'}
            className="px-6 py-2 bg-orange-600 text-white font-medium rounded-lg hover:bg-orange-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {metaMaskStatus === 'not_installed' ? 'Install MetaMask First' : 'Connect MetaMask'}
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Smart Contract Address *
            </label>
            <input
              type="text"
              value={config.contract_address}
              onChange={(e) => setConfig(prev => ({ ...prev, contract_address: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="0x..."
              required
            />
            <p className="text-xs text-gray-500 mt-1">Deploy SecureVoting.sol on Sepolia using Remix IDE and enter the address here</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Private Key *
            </label>
            <input
              type="password"
              value={config.private_key}
              onChange={(e) => setConfig(prev => ({ ...prev, private_key: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="Your wallet private key"
              required
            />
            <p className="text-xs text-red-500 mt-1">⚠️ Export from MetaMask: Account Details → Export Private Key</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Account Address
            </label>
            <input
              type="text"
              value={config.account_address}
              onChange={(e) => setConfig(prev => ({ ...prev, account_address: e.target.value }))}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-50"
              placeholder="Auto-filled from MetaMask"
              readOnly
            />
            <p className="text-xs text-gray-500 mt-1">Automatically filled when MetaMask is connected</p>
          </div>

          <button
            type="submit"
            disabled={isLoading || metaMaskStatus !== 'connected'}
            className="w-full px-6 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Configuring...' : 'Configure Blockchain'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default BlockchainConfig;
