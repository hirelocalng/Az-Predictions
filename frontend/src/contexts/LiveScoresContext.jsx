import { createContext, useContext } from 'react'

export const LiveScoresContext = createContext({})
export const useLiveScores = () => useContext(LiveScoresContext)
