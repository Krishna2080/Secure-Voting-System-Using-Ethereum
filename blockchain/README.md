
# Blockchain Integration with MetaMask & Sepolia

This guide shows how to integrate the voting system with Ethereum blockchain using **only** MetaMask and Remix IDE.

## Prerequisites

1. **MetaMask Extension**: Install MetaMask browser extension
2. **Sepolia Testnet**: Configure MetaMask to use Sepolia testnet
3. **Test ETH**: Get Sepolia ETH from faucets for gas fees
4. **Remix IDE**: Use Remix IDE for contract deployment

## Step-by-Step Setup

### 1. Install MetaMask
- Go to metamask.io and install the browser extension
- Create a new wallet or import existing one
- **Important**: Save your seed phrase securely

### 2. Configure Sepolia Testnet
- Open MetaMask
- Click on network dropdown (usually shows "Ethereum Mainnet")
- Select "Sepolia test network"
- If not available, add manually:
  - Network Name: Sepolia Test Network
  - RPC URL: https://rpc.sepolia.org
  - Chain ID: 11155111
  - Currency Symbol: ETH
  - Block Explorer: https://sepolia.etherscan.io

### 3. Get Test ETH
Get Sepolia ETH from these faucets:
- https://sepolia-faucet.pk910.de/ (Recommended)
- https://sepoliafaucet.com/
- https://faucets.chain.link/sepolia

You'll need test ETH to deploy contracts and pay for transactions.

### 4. Deploy Smart Contract using Remix

1. **Open Remix IDE**: Go to https://remix.ethereum.org
2. **Create Contract**: Create a new file called `SecureVoting.sol`
3. **Copy Contract Code**: Copy the contract code from `VotingContract.sol`
4. **Compile Contract**:
   - Go to "Solidity Compiler" tab
   - Select compiler version 0.8.19 or later
   - Click "Compile SecureVoting.sol"
5. **Deploy Contract**:
   - Go to "Deploy & Run Transactions" tab
   - Set Environment to "Injected Provider - MetaMask"
   - Make sure MetaMask is connected and on Sepolia
   - In constructor, add candidate names: `["candidate1", "candidate2", "candidate3"]`
   - Click "Deploy"
   - Confirm transaction in MetaMask
6. **Copy Contract Address**: After deployment, copy the contract address

### 5. Configure Backend

1. **Start Backend**: Run the Python backend with `python main.py`
2. **Open Admin Panel**: Go to http://localhost:8080 and click "Admin Portal"
3. **Configure Blockchain**:
   - Click "Configure Blockchain"
   - Click "Connect MetaMask" to auto-fill wallet details
   - Enter your contract address from step 4
   - Export your private key from MetaMask:
     - Click on account menu (3 dots)
     - Select "Account Details"
     - Click "Export Private Key"
     - Enter MetaMask password
     - Copy the private key (keep it secure!)
   - Paste the private key in the configuration form
   - Click "Configure Blockchain"

### 6. Start Voting

1. **Register Voters**: Use face recognition to register voters
2. **Cast Votes**: Authenticated voters can cast votes
3. **View Results**: Results are stored both locally and on blockchain

## Smart Contract Features

- **Immutable Voting**: All votes are permanently recorded on Sepolia
- **Fraud Prevention**: Each voter can only vote once
- **Transparency**: All transactions are publicly viewable on Sepolia Etherscan
- **Security**: Uses MetaMask's secure transaction signing

## Verification & Transparency

After deployment, verify your contract on Sepolia Etherscan:
1. Go to https://sepolia.etherscan.io
2. Search for your contract address
3. View all transactions and contract interactions
4. Verify voting transparency and immutability

## Troubleshooting

**MetaMask Connection Issues:**
- Ensure MetaMask is unlocked
- Check you're on Sepolia testnet
- Refresh the page and try again

**Transaction Failures:**
- Check you have sufficient Sepolia ETH
- Increase gas limit if needed
- Verify contract address is correct

**Backend Connection:**
- Ensure Python backend is running on localhost:8000
- Check frontend is running on localhost:8080
- Verify no firewall is blocking connections

**Contract Deployment Issues:**
- Make sure you're connected to Sepolia in Remix
- Check you have enough Sepolia ETH for gas
- Verify contract compiles without errors

## Gas Costs

Typical gas costs on Sepolia:
- Contract deployment: ~500,000-1,000,000 gas
- Vote casting: ~50,000-80,000 gas
- Vote checking: ~21,000 gas (read-only)

With current gas prices, each vote costs approximately 0.001-0.002 Sepolia ETH.

## Security Notes

- **Private Keys**: Never share your private key
- **Testnet Only**: This setup is for Sepolia testnet only
- **Production**: For mainnet, use hardware wallets or secure key management
- **Contract Immutability**: Smart contracts cannot be modified once deployed

## Architecture

```
Frontend (React) ↔ Backend (Python) ↔ Blockchain (Sepolia)
     ↓                    ↓                    ↓
  Face Auth         Local Storage      Smart Contract
  MetaMask         Vote Backup         Immutable Votes
```

This ensures votes are recorded both locally (for speed) and on blockchain (for immutability).
