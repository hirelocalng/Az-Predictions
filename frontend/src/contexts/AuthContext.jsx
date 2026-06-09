import { createContext, useContext } from 'react'
export const AuthContext = createContext({ auth: null, isPremium: false, navigate: () => {}, logout: () => {} })
export const useAuth = () => useContext(AuthContext)
